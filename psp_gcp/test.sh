# while getopts abcde opt; do
#     case $opt in
#         a) flag1="SET"
#         ;;
#         b) flag2="SET"
#         ;;
#         c) optflag1="SET"
#         ;;
#         d) optflag2="SET"
#         ;;
#         e) optflag3="SET"
#         ;;
#     esac
# done
#
# echo $flag1
# echo $flag2
# echo $flag3

#!/bin/bash
#
# while getopts ":long:c:" opt; do
#   case $opt in
#     long) arg_1="$OPTARG"
#     ;;
#     p) p_out="$OPTARG"
#     ;;
#     \?) echo "Invalid option -$OPTARG" >&2
#     ;;
#   esac
# done
#
# if [ -z "$p_out" ]; then p_out=42; fi
# if [ -z "$arg_1" ]; then p_out=500; fi
#
#
# printf "Argument p_out is %s\n" "$p_out"
# printf "Argument arg_1 is %s\n" "$arg_1"
# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
    "--help") set -- "$@" "-h" ;;
    "--rest") set -- "$@" "-r" ;;
    "--ws")   set -- "$@" "-w" ;;
    *)        set -- "$@" "$arg"
  esac
done

# Default behavior
rest=false; ws=false

# Parse short options
OPTIND=1
while getopts "hrw" opt
do
  case "$opt" in
    "h") if [-z $OPTARG]; then echo "Using default params"; exit 0 ;;
    "r") echo "REST" ;;
    "w") ws=true ;;
    "?") echo "Using default params"; exit 1 ;;
  esac
done
shift $(expr $OPTIND - 1) # remove options from positional parameters
