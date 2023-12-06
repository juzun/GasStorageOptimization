/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the class library                   */
/*       SoPlex --- the Sequential object-oriented simPlex.                  */
/*                                                                           */
/*  Copyright (c) 1996-2023 Zuse Institute Berlin (ZIB)                      */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*      http://www.apache.org/licenses/LICENSE-2.0                           */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*  You should have received a copy of the Apache-2.0 license                */
/*  along with SoPlex; see the file LICENSE. If not email to soplex@zib.de.  */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxsolver.h"
#include "soplex/spxfileio.h"

namespace soplex
{
template <class R>
bool SPxSolverBase<R>::readBasisFile(
   const char*    filename,
   const NameSet* rowNames,
   const NameSet* colNames)
{
   spxifstream file(filename);

   if(!file)
      return false;

   return this->readBasis(file, rowNames, colNames);
}

template <class R>
bool SPxSolverBase<R>::writeBasisFile
(const char*    filename,
 const NameSet* rowNames,
 const NameSet* colNames,
 const bool cpxFormat) const
{
   std::ofstream file(filename);

   if(!file)
      return false;

   this->writeBasis(file, rowNames, colNames);
   return true;
}

} // namespace soplex