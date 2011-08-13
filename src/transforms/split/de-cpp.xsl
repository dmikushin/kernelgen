<?xml version="1.0" encoding="ISO-8859-1"?>

<!--

 kernelgen - an XSLT-based Fortran source to source preprocessor.
 
 This file is part of kernelgen.
 
 (c) 2009, 2011 Dmitry Mikushin
 
 kernelgen is a free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Softawre Foundation; either version 2 of the License, or
 (at your option) any later version.
 
 kernelgen is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with md5sum; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

 Purpose: Remove preprocessor tags <cpp> and <unseen>. Not sure if it is
 a good idea, as we in fact have the preprocessing done. There is likely
 no better choise for now, or alternatively we will need to apply all
 transformations with respect to preprocessor variants. Additionally
 remove Fortran include statement, that works technically similar to
 preprocessor's #include.

-->

<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:F="http://g95-xml.sourceforge.net/">

<xsl:template match="F:cpp">
</xsl:template>

<xsl:template match="F:_cpp-include_">
</xsl:template>

<xsl:template match="F:unseen">
</xsl:template>

<xsl:template match="F:_fortran-include_">
</xsl:template>

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/> 
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>
