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

 Purpose: In each individual definition mark definied symbol
 and group of symbols, used in type declaration.

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

<xsl:template match="F:entity-decl" mode="process">
  <xsl:for-each select=".//F:s">
    <xsl:element name="s" namespace="http://g95-xml.sourceforge.net/">
      <xsl:attribute name="N">
        <xsl:value-of select="@N"/>
      </xsl:attribute>
      <xsl:attribute name="class">
        <xsl:choose>
<!--
Symbol is a declarated object.
-->
          <xsl:when test="name(..) = &quot;_obj-N_&quot;">
            <xsl:text>object</xsl:text>
          </xsl:when>
          <xsl:when test="name(..) = &quot;_fct_inl_&quot;">
            <xsl:text>object</xsl:text>
          </xsl:when>
          <xsl:when test="name(..) = &quot;_T-N_&quot;">
            <xsl:text>object</xsl:text>
          </xsl:when>
<!--
Otherwise, symbol is a dependency.
-->
          <xsl:otherwise>
            <xsl:text>dependency</xsl:text>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:attribute>
<!-- 
Every symbol tag to carry an attribute, indentifying the type of symbol:
function, subroutine, parameter or variable.
-->
      <xsl:variable name="is-parameter">
        <xsl:for-each select="../../../F:gforscale-decl-body/F:attr-spec-lst/F:attr-spec">
          <xsl:if test="@N = &quot;parameter&quot;">
            <xsl:text>1</xsl:text>
          </xsl:if>
        </xsl:for-each>
      </xsl:variable>
      <xsl:variable name="type">
        <xsl:choose>
          <xsl:when test="name(..) = &quot;_subr_&quot;">
            <xsl:text>subroutine</xsl:text>
          </xsl:when>
          <xsl:when test="name(..) = &quot;_fct_&quot;">
            <xsl:text>function</xsl:text>
          </xsl:when>
          <xsl:when test="name(..) = &quot;_fct_inl_&quot;">
            <xsl:text>function</xsl:text>
          </xsl:when>
          <xsl:when test="name(..) = &quot;_T-N_&quot;">
            <xsl:text>type</xsl:text>
          </xsl:when>
          <xsl:when test="$is-parameter = &quot;1&quot;">
            <xsl:text>parameter</xsl:text>
          </xsl:when>
          <xsl:otherwise>
            <xsl:text>variable</xsl:text>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:variable>      
      <xsl:attribute name="type">
        <xsl:value-of select="$type"/>
      </xsl:attribute>
      <xsl:value-of select="."/>
    </xsl:element>
  </xsl:for-each>
</xsl:template>

<xsl:template match="F:s" mode="symbols-filter">
  <xsl:element name="s" namespace="http://g95-xml.sourceforge.net/">
    <xsl:attribute name="N">
      <xsl:value-of select="@N"/>
    </xsl:attribute>
    <xsl:attribute name="class">
      <xsl:text>dependency</xsl:text>
    </xsl:attribute>
    <xsl:value-of select="."/>
  </xsl:element>
</xsl:template>

<xsl:template match="F:gforscale-decl-body" mode="process">
<!--
In type declaration section every used symbol
is a dependency symbol.
-->
  <xsl:apply-templates select=".//F:s" mode="symbols-filter"/>
</xsl:template>

<xsl:template match="F:T-decl-stmt">
  <xsl:element name="T-decl-stmt" namespace="http://g95-xml.sourceforge.net/">
    <xsl:element name="symbols" namespace="http://g95-xml.sourceforge.net/">
       <xsl:apply-templates match=".//F:entity-decl|.//F:gforscale-decl-body" mode="process"/>
    </xsl:element>
    <xsl:apply-templates match="."/>
  </xsl:element>
</xsl:template>

</xsl:stylesheet>
