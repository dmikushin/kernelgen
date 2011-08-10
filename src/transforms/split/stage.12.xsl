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

 Purpose: Mark globally defined symbols.

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

<xsl:template match="F:symbols">
  <xsl:element name="symbols" namespace="http://g95-xml.sourceforge.net/">
    <xsl:for-each select=".//F:s">
      <xsl:variable name="symbol-name">
        <xsl:value-of select="@N"/>
      </xsl:variable>
<!--
Check symbol is not a type name.
-->
      <xsl:if test="(count(../../F:user-T-lst/F:l-G[@N = $symbol-name]) = 0)">
<!--
Check if symbol is defined private
(and do not filter out local private symbols).
-->
      <xsl:variable name="private-local-defined-symbol">
        <xsl:for-each select="../../F:definitions/F:T-decl-stmt">
          <xsl:if test="F:entity-decl/F:_obj-N_/F:s/@N = $symbol-name">
            <xsl:for-each select=".//F:gforscale-decl-body/F:attr-spec-lst/F:attr-spec">
              <xsl:if test="@N = &quot;private&quot;">
                <xsl:text>1</xsl:text>
              </xsl:if>
            </xsl:for-each>
          </xsl:if>
        </xsl:for-each>
      </xsl:variable>
<!--
Check if symbol is global.
-->
      <xsl:variable name="global-defined-symbol">
        <xsl:variable name="global-defined-symbol-name">
          <xsl:value-of select="@N"/>
        </xsl:variable>
<!--
Global symbol criteria: symbol is in global list, and
there is up to 1 local definition (otherwise, if symbol is defined
twice, then one definition may be originated from module, and
another one is from local scope - in this case symbol is local).
-->
        <xsl:if test="((count(../../F:S-lst/F:l-G[@N = $symbol-name]) > 0) and
(count(../../F:definitions/F:T-decl-stmt[(@global = 0) and
(F:entity-decl/F:_obj-N_/F:s/@N = $symbol-name)]) = 0))">
          <xsl:text>1</xsl:text>
        </xsl:if>
      </xsl:variable>
<!--
Create symbol node with global attribute.
-->
      <xsl:element name="s" namespace="http://g95-xml.sourceforge.net/">
        <xsl:attribute name="N">
          <xsl:value-of select="@N"/>
        </xsl:attribute>
        <xsl:choose>
          <xsl:when test="(($private-local-defined-symbol = &quot;1&quot;) and ($global-defined-symbol = &quot;1&quot;)) or ($global-defined-symbol != &quot;1&quot;)">
            <xsl:attribute name="global">
              <xsl:value-of select="0"/>
            </xsl:attribute>
<!--
Check with symbol definition,
if it is allocatable.
-->
            <xsl:attribute name="allocatable">
              <xsl:choose>
                <xsl:when test="../../F:definitions/F:T-decl-stmt[F:entity-decl/F:_obj-N_/F:s/@N = $symbol-name]/F:gforscale-decl-body/F:attr-spec-lst/F:attr-spec[@N = &quot;allocatable&quot;]">
                  <xsl:value-of select="1"/>
                </xsl:when>
                <xsl:otherwise>
                  <xsl:value-of select="0"/>
                </xsl:otherwise>
              </xsl:choose>
            </xsl:attribute>
          </xsl:when>
          <xsl:otherwise>
            <xsl:attribute name="global">
              <xsl:value-of select="1"/>
            </xsl:attribute>
<!--
Copy additional attributes from
global symbol definition.
-->
            <xsl:variable name="local-name">
              <xsl:value-of select="@N"/>
            </xsl:variable>
            <xsl:variable name="global-name">
              <xsl:value-of select="../../F:S-lst/F:l-G[@N = $local-name]/@G-N"/>
            </xsl:variable>
            <xsl:attribute name="allocatable">
              <xsl:value-of select="../../F:G-S-lst/F:G-var[@N = $global-name]/@allocatable"/>
            </xsl:attribute>
            <xsl:attribute name="T">
              <xsl:value-of select="../../F:G-S-lst/F:G-var[@N = $global-name]/@T"/>
            </xsl:attribute>
            <xsl:attribute name="kind">
              <xsl:value-of select="../../F:G-S-lst/F:G-var[@N = $global-name]/@kind"/>
            </xsl:attribute>
            <xsl:attribute name="dimension">
              <xsl:value-of select="../../F:G-S-lst/F:G-var[@N = $global-name]/@dimension"/>
            </xsl:attribute>
            <xsl:attribute name="derived">
              <xsl:value-of select="../../F:G-S-lst/F:G-var[@N = $global-name]/@derived"/>
            </xsl:attribute>
          </xsl:otherwise>
        </xsl:choose>
        <xsl:attribute name="type">
          <xsl:value-of select="@type"/>
        </xsl:attribute>
      </xsl:element>
      </xsl:if>
    </xsl:for-each>
  </xsl:element>
</xsl:template>

</xsl:stylesheet>
