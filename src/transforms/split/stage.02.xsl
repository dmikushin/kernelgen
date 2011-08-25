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

 Purpose: Split grouped definitions into individual.

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

<xsl:template match="F:T-decl-stmt/F:entity-decl-lst"/>

<xsl:template match="node()|@*" mode="private">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*" mode="private"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="F:T-decl-stmt/F:entity-decl-lst" mode="private"/>

<xsl:template match="F:attr-spec-lst" mode="private">
  <xsl:element name="attr-spec-lst" namespace="http://g95-xml.sourceforge.net/">
    <xsl:copy-of select=".//F:attr-spec"/>
    <xsl:if test="count(.//F:attr-spec) > 0">
      <xsl:element name="c" namespace="http://g95-xml.sourceforge.net/">
        <xsl:text>,&#32;</xsl:text>
      </xsl:element>
    </xsl:if>
    <xsl:element name="attr-spec" namespace="http://g95-xml.sourceforge.net/">
      <xsl:attribute name="N">
        <xsl:text>private</xsl:text>
      </xsl:attribute>
    </xsl:element>
  </xsl:element>
</xsl:template>

<xsl:template match="F:T-decl-stmt">
  <xsl:variable name="type">
    <xsl:apply-templates match=".//F:kernelgen-decl-body"/>
  </xsl:variable>
  <xsl:variable name="type-private">
    <xsl:apply-templates match=".//F:kernelgen-decl-body" mode="private"/>
  </xsl:variable>
  <xsl:for-each select=".//F:entity-decl">
<!--
Get the name of defined symbol.
-->
    <xsl:variable name="symbol">
      <xsl:value-of select="F:_obj-N_/F:s/@N"/>
    </xsl:variable>
<!--
Check if the symbol is declared PRIVATE
by the standalone private-stmt.
-->
    <xsl:variable name="private">
      <xsl:for-each select="../../../F:private-stmt">
        <xsl:for-each select=".//F:s">
          <xsl:if test="@N = $symbol">
            <xsl:text>1</xsl:text>
          </xsl:if>
        </xsl:for-each>
      </xsl:for-each>
    </xsl:variable>
    <xsl:element name="T-decl-stmt" namespace="http://g95-xml.sourceforge.net/">
      <xsl:choose>
        <xsl:when test="$private != &quot;&quot;">
          <xsl:copy-of select="$type-private"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:copy-of select="$type"/>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:copy-of select="."/>
      <xsl:element name="L" namespace="http://g95-xml.sourceforge.net/">
        <xsl:text>&#10;</xsl:text>
      </xsl:element>
    </xsl:element>
  </xsl:for-each>
</xsl:template>

</xsl:stylesheet>
