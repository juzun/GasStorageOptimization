#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "libzimpl" for configuration "Release"
set_property(TARGET libzimpl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libzimpl PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libzimpl.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libzimpl )
list(APPEND _IMPORT_CHECK_FILES_FOR_libzimpl "${_IMPORT_PREFIX}/lib/libzimpl.a" )

# Import target "zimpl" for configuration "Release"
set_property(TARGET zimpl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(zimpl PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/zimpl"
  )

list(APPEND _IMPORT_CHECK_TARGETS zimpl )
list(APPEND _IMPORT_CHECK_FILES_FOR_zimpl "${_IMPORT_PREFIX}/bin/zimpl" )

# Import target "libzimpl-pic" for configuration "Release"
set_property(TARGET libzimpl-pic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libzimpl-pic PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libzimpl-pic.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libzimpl-pic )
list(APPEND _IMPORT_CHECK_FILES_FOR_libzimpl-pic "${_IMPORT_PREFIX}/lib/libzimpl-pic.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
