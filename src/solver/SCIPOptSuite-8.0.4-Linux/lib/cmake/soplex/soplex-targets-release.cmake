#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "soplex" for configuration "Release"
set_property(TARGET soplex APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(soplex PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/soplex"
  )

list(APPEND _IMPORT_CHECK_TARGETS soplex )
list(APPEND _IMPORT_CHECK_FILES_FOR_soplex "${_IMPORT_PREFIX}/bin/soplex" )

# Import target "libsoplex" for configuration "Release"
set_property(TARGET libsoplex APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libsoplex PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsoplex.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libsoplex )
list(APPEND _IMPORT_CHECK_FILES_FOR_libsoplex "${_IMPORT_PREFIX}/lib/libsoplex.a" )

# Import target "libsoplex-pic" for configuration "Release"
set_property(TARGET libsoplex-pic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libsoplex-pic PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsoplex-pic.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libsoplex-pic )
list(APPEND _IMPORT_CHECK_FILES_FOR_libsoplex-pic "${_IMPORT_PREFIX}/lib/libsoplex-pic.a" )

# Import target "libsoplexshared" for configuration "Release"
set_property(TARGET libsoplexshared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libsoplexshared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsoplexshared.so.6.0.4.0"
  IMPORTED_SONAME_RELEASE "libsoplexshared.so.6.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS libsoplexshared )
list(APPEND _IMPORT_CHECK_FILES_FOR_libsoplexshared "${_IMPORT_PREFIX}/lib/libsoplexshared.so.6.0.4.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
