if(NOT TARGET libzimpl)
  include("${CMAKE_CURRENT_LIST_DIR}/zimpl-targets.cmake")
endif()

set(ZIMPL_LIBRARIES libzimpl)
set(ZIMPL_PIC_LIBRARIES libzimpl-pic)
set(ZIMPL_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(ZIMPL_FOUND TRUE)

