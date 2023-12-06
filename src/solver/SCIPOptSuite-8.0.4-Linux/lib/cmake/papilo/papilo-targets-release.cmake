#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "papilo-core" for configuration "Release"
set_property(TARGET papilo-core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(papilo-core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libpapilo-core.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS papilo-core )
list(APPEND _IMPORT_CHECK_FILES_FOR_papilo-core "${_IMPORT_PREFIX}/lib/libpapilo-core.a" )

# Import target "clusol" for configuration "Release"
set_property(TARGET clusol APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clusol PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclusol.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS clusol )
list(APPEND _IMPORT_CHECK_FILES_FOR_clusol "${_IMPORT_PREFIX}/lib/libclusol.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
