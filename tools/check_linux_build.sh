#!/usr/bin/env bash
# Builds a commit of this repository from scratch on a Linux machine, so that a
# change made and tested on macOS is known to compile where the OpenGL backend
# lives. A Linux build broke once already without anyone noticing, because
# nothing built it until the next test there.
#
#   tools/check_linux_build.sh [revision] [--verify]
#
# The revision defaults to HEAD. What is built is the commit, through
# git archive, not the working tree, so uncommitted edits neither help nor
# hurt it.
#
# The plug-in is built twice, because its OpenGL path compiles differently
# depending on the Ogre it finds: against the patched rviz_ogre_vendor, which
# can take an instance count from the GPU, and against the stock one in the
# ROS distribution, which is what anyone without the patch has. Each build
# starts from an empty build and install directory, so nothing cached from an
# earlier build can hide an error.
#
# --verify also runs the OpenGL verifier on the patched build, headless
# through EGL. It needs the machine's GPU, but no display. It is given a time
# limit: on 2026-10-06 Shannon's driver blocked every EGL context in the kernel
# (nvkms_open_common) while its X sessions held the display, binaries that had
# passed included, and the check sat there for ten minutes.
#
# The machine and paths can be changed through the environment:
#   GSPLAT_LINUX_HOST          ssh destination        (shannon-ts)
#   GSPLAT_LINUX_WORKSPACE     workspace, under ~     (ros2_ws/gsplat-check)
#   GSPLAT_LINUX_PATCHED_OGRE  patched rviz_ogre_vendor install, under ~
#                                                     (ros2_ws/install/rviz_ogre_vendor)
#   GSPLAT_LINUX_ROS_SETUP     the ROS distribution   (/opt/ros/humble/setup.bash)
#   GSPLAT_LINUX_VERIFY_TIMEOUT  seconds for --verify  (120)
set -euo pipefail

host=${GSPLAT_LINUX_HOST:-shannon-ts}
workspace=${GSPLAT_LINUX_WORKSPACE:-ros2_ws/gsplat-check}
patched_ogre=${GSPLAT_LINUX_PATCHED_OGRE:-ros2_ws/install/rviz_ogre_vendor}
ros_setup=${GSPLAT_LINUX_ROS_SETUP:-/opt/ros/humble/setup.bash}
verify_timeout=${GSPLAT_LINUX_VERIFY_TIMEOUT:-120}

revision=HEAD
verify=0
for argument in "$@"; do
  case "$argument" in
    --verify) verify=1 ;;
    -*) echo "unknown option: $argument" >&2; exit 2 ;;
    *) revision=$argument ;;
  esac
done

repository=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
commit=$(git -C "$repository" rev-parse --short "$revision")
echo "Building $commit ($(git -C "$repository" log -1 --format=%s "$revision"))"
echo "on $host in ~/$workspace"

# The workspace sits beside the ROS workspace on that machine, where a colcon
# run from ~/ros2_ws would otherwise find its sources and fail on the
# duplicate package names. COLCON_IGNORE keeps them out of that search.
ship='set -eu
workspace=$HOME/$1
rm -rf "${workspace:?}/src"
mkdir -p "$workspace/src"
touch "$workspace/COLCON_IGNORE"
tar -x -C "$workspace/src"'
git -C "$repository" archive --format=tar "$revision" \
    gaussian_splatting_msgs gaussian_splatting_rviz_plugins |
  ssh "$host" "bash -c $(printf '%q' "$ship") _ $(printf '%q' "$workspace")"

ssh "$host" bash -s -- \
    "$(printf '%q' "$workspace")" "$(printf '%q' "$patched_ogre")" \
    "$(printf '%q' "$ros_setup")" "$verify" "$verify_timeout" <<'REMOTE'
set -uo pipefail
workspace=$HOME/$1
patched_ogre=$HOME/$2
ros_setup=$3
verify=$4
verify_timeout=$5
cd "$workspace"

failed=0
for variant in patched stock; do
  rm -rf "${workspace:?}/build-$variant" "${workspace:?}/install-$variant" \
    "${workspace:?}/log-$variant"
  log=$workspace/build-$variant.log
  start=$SECONDS
  (
    # The setup scripts read variables they never set.
    set +u
    source "$ros_setup"
    if [ "$variant" = patched ]; then
      source "$patched_ogre/share/rviz_ogre_vendor/local_setup.bash"
    fi
    colcon --log-base "log-$variant" build --base-paths src \
      --build-base "build-$variant" --install-base "install-$variant" \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      --event-handlers console_direct+
  ) > "$log" 2>&1
  status=$?
  seconds=$((SECONDS - start))

  flags=$workspace/build-$variant/gaussian_splatting_rviz_plugins/CMakeFiles/gaussian_splatting_rviz_plugins.dir/flags.make
  if [ -f "$flags" ] && grep -q GSPLAT_OGRE_GL_INDIRECT_DRAW "$flags"; then
    indirect="GPU instance count"
  else
    indirect="no GPU instance count"
  fi
  if [ $status -eq 0 ]; then
    echo "$variant Ogre: built in ${seconds}s ($indirect)"
  else
    echo "$variant Ogre: FAILED after ${seconds}s ($indirect); log in $log"
    failed=1
    # Compiler and linker errors first; failing that, the end of the log.
    if ! grep -E "error:|undefined reference|CMake Error" "$log" | sort -u | head -30; then
      tail -30 "$log"
    fi
  fi
  # Warnings in this repository's own sources, not in the headers it includes.
  warnings=$(grep -E "^$workspace/src/[^:]+:[0-9]+:[0-9]+: warning:" "$log" | sort -u || true)
  if [ -n "$warnings" ]; then
    echo "  $(printf '%s\n' "$warnings" | wc -l) warning(s) in this repository's sources:"
    printf '%s\n' "$warnings" | sed "s|^$workspace/src/|    |" | head -20
  fi
done

if [ "$verify" = 1 ]; then
  verifier=$workspace/build-patched/gaussian_splatting_rviz_plugins/verify_gl_preparation
  shader=$workspace/src/gaussian_splatting_rviz_plugins/ogre_media/materials/programs/GLSL430/gsplat_prepare.comp
  if [ -x "$verifier" ]; then
    echo "OpenGL verifier, patched build:"
    timeout --kill-after=5 "$verify_timeout" "$verifier" --quick "$shader" \
      < /dev/null > "$workspace/verify.log" 2>&1
    code=$?
    if [ $code -eq 77 ]; then
      echo "  skipped: no OpenGL 4.3 context through EGL"
    elif [ $code -eq 124 ] || [ $code -eq 137 ]; then
      echo "  TIMED OUT after ${verify_timeout}s with no result; the build is not at"
      echo "  fault if a verifier that passed before hangs too. Where it waits:"
      echo "  run it in the background and read ps -o wchan= -p <pid>."
      failed=1
    else
      grep -E "failing check|FAIL" "$workspace/verify.log" | head -20
      [ $code -eq 0 ] || failed=1
    fi
  else
    echo "OpenGL verifier: not built (EGL or GLEW missing)"
    failed=1
  fi
fi
exit $failed
REMOTE
