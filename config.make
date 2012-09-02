# add custom variables to this file

# OF_ROOT allows to move projects outside apps/* just set this variable to the
# absoulte path to the OF root folder

OF_ROOT = ../openFrameworks

# USER_CFLAGS allows to pass custom flags to the compiler
# for example search paths like:
# USER_CFLAGS = -I src/objects

USER_CFLAGS = -g -I ./include \
-I ./include/OpenclPlatform -I /usr/include/eigen3 -I /usr/local/cuda/include \
-I ~/NVIDIA_GPU_Computing_SDK/C/common/inc \

USER_CFLAGS_CUDA = -I ./include -I /home/sk/NVIDIA_GPU_Computing_SDK/C/common/inc \
-arch=sm_21

# USER_LDFLAGS allows to pass custom flags to the linker
# for example libraries like:
# USER_LD_FLAGS = libs/libawesomelib.a

USER_LDFLAGS =

# use this to add system libraries for example:
# USER_LIBS = -lpango
 
USER_LIBS = -lusb-1.0 -lX11 -lcuda -L /usr/local/cuda/lib64 -lcudart

# change this to add different compiler optimizations to your project

USER_COMPILER_OPTIMIZATION = -march=native -mtune=native -O3

EXCLUDE_FROM_SOURCE="bin,.xcodeproj,obj"
