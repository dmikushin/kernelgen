<?xml version="1.0" encoding="ISO-8859-1"?>

<!--

 gforscale - an XSLT-based Fortran source to source preprocessor.
 
 This file is part of gforscale.
 
 (c) 2009, 2011 Dmitry Mikushin
 
 gforscale is a free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Softawre Foundation; either version 2 of the License, or
 (at your option) any later version.
 
 gforscale is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with md5sum; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

 Purpose: For each do-construct symbol in symbols list,
 add dependencies for their definitions and mark used definitions.

-->

<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:F="http://g95-xml.sourceforge.net/"
xmlns:exsl="http://exslt.org/common" extension-element-prefixes="exsl">

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/> 
  </xsl:copy>
</xsl:template> 

<xsl:template match="F:definitions">
</xsl:template>

<xsl:template match="F:T-decl-stmt" mode="process">
  <xsl:param name="name"/>
  <xsl:param name="type"/>
  <xsl:variable name="T-decl-stmt">
    <xsl:copy-of select="exsl:node-set(.)"/>
  </xsl:variable>
  <xsl:variable name="symbol-dependencies">
<!--
Select only those symbols that are subject to the current declaration statement and
match the specified name.
-->
    <xsl:for-each select=".//F:symbols/F:s[@N = $name and @class = &quot;object&quot;]">
      <xsl:copy-of select="exsl:node-set(.)"/>
<!--
Adding marker with text to make variable non-empty (will be filtered out later).
-->
      <xsl:element name="gforscale-depsyms" namespace="http://g95-xml.sourceforge.net/">
        <xsl:text>&#32;</xsl:text>
      </xsl:element>
    </xsl:for-each>
  </xsl:variable>
<!--
OK, let T-decl-stmt to be inside symbols list,
it will be moved out in next stage.
-->
  <xsl:if test="not($symbol-dependencies = &quot;&quot;)">
<!--
If symbol is not a subroutine (variable or function), write dependencies
(including itself). Used for prototype list formed on future stages.
-->
    <xsl:if test="not($type = &quot;subroutine&quot;)">
      <xsl:for-each select="exsl:node-set($symbol-dependencies)/F:s">
        <xsl:copy-of select="."/>
      </xsl:for-each>
    </xsl:if>
<!--
Recursively perform the same dependencies lookup rule
for symbols in entire declaration statement (will need to
filter duplicates on later steps).
-->
    <xsl:for-each select=".//F:symbols/F:s[@class = &quot;dependency&quot;]">
      <xsl:variable name="name">
        <xsl:value-of select="@N"/>
      </xsl:variable>
      <xsl:choose>
<!--
If symbol is not defined locally (e.g. a global symbol), leave it alone.
-->
        <xsl:when test="count(../../../F:T-decl-stmt[count(F:symbols/F:s[(@N = $name) and (@class = &quot;object&quot;)]) = 1]) = 0">
          <xsl:copy-of select="."/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:apply-templates select="../../../F:T-decl-stmt" mode="process">
            <xsl:with-param name="name">
              <xsl:value-of select="@N"/>
            </xsl:with-param>
            <xsl:with-param name="type">
              <xsl:value-of select="@type"/>
            </xsl:with-param>
          </xsl:apply-templates>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>
<!--
Add entire declaration statement to the kernel loop routine's
list of declarations.
-->
    <xsl:copy-of select="$T-decl-stmt"/>
  </xsl:if>
</xsl:template>

<xsl:template match="F:do-group/F:symbols">
  <xsl:element name="symbols" namespace="http://g95-xml.sourceforge.net/">
    <xsl:for-each select=".//F:s">
      <xsl:variable name="name">
        <xsl:value-of select="@N"/>
      </xsl:variable>
      <xsl:choose>
<!--
If symbol is not defined locally (e.g. a global symbol), leave it alone.
-->
        <xsl:when test="count(../../F:definitions/F:T-decl-stmt[count(F:symbols/F:s[(@N = $name) and (@class = &quot;object&quot;)]) = 1]) = 0">
          <xsl:copy-of select="."/>
        </xsl:when>
<!--
Look up for a symbol definition.
-->
        <xsl:otherwise>
          <xsl:apply-templates select="../../F:definitions/F:T-decl-stmt" mode="process">
            <xsl:with-param name="name">
              <xsl:value-of select="@N"/>
            </xsl:with-param>
            <xsl:with-param name="type">
              <xsl:value-of select="@type"/>
            </xsl:with-param>
          </xsl:apply-templates>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>
  </xsl:element>
</xsl:template>
 
</xsl:stylesheet>
