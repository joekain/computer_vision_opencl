add_executable (main main.cxx)
target_include_directories (main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (main ${OpenCL_LIBRARY} ${OpenCV_LIBS})
