bpf_object(gpuevent_snoop gpuevent_snoop/bpf/gpuevent_snoop.bpf.c)
add_dependencies(gpuevent_snoop_skel libbpf-build bpftool-build)

set(GPU_EVENT_SOURCES
    gpuevent_snoop/GpuEventSnoop.cpp
)

set(GPU_EVENT_HEADERS
    gpuevent_snoop/bpf/gpuevent_snoop.h
    gpuevent_snoop.skel.h
)

add_executable(gpuevent_snoop ${GPU_EVENT_SOURCES} ${GPU_EVENT_HEADERS})

target_include_directories(gpuevent_snoop PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_link_libraries(gpuevent_snoop PRIVATE gpuevent_snoop_skel fmt utils)
