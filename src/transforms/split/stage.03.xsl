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

 Purpose: Mark definitions matching to inline functions,
 transform inline functions into new definitons.

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

<xsl:template match="F:stmt-fct-stmt">
  <xsl:element name="T-decl-stmt" namespace="http://g95-xml.sourceforge.net/">
    <xsl:variable name="name">
      <xsl:value-of select=".//F:_stmt-fct_/F:s/@N"/>
    </xsl:variable>
    <xsl:element name="kernelgen-decl-body" namespace="http://g95-xml.sourceforge.net/">
<!--
Put original statement plain code here.
-->
      <xsl:element name="c" namespace="http://g95-xml.sourceforge.net/">
        <xsl:value-of select="."/>
        <xsl:text>&#10;</xsl:text>
      </xsl:element>
<!--
From symbols take only tags, no content.
-->
      <xsl:for-each select=".//F:s[@N != $name]">
        <xsl:element name="s" namespace="http://g95-xml.sourceforge.net/">
          <xsl:attribute name="N">
            <xsl:value-of select="@N"/>
          </xsl:attribute>
        </xsl:element>
      </xsl:for-each>
    </xsl:element>
<!--
In function definition statement insert fake symbol
to pull function definition into kernel routines.
-->
    <xsl:element name="entity-decl" namespace="http://g95-xml.sourceforge.net/">
      <xsl:element name="_fct_inl_" namespace="http://g95-xml.sourceforge.net/">
        <xsl:element name="s" namespace="http://g95-xml.sourceforge.net/">
          <xsl:attribute name="N">
            <xsl:value-of select="$name"/>
          </xsl:attribute>
        </xsl:element>
      </xsl:element>
    </xsl:element>
  </xsl:element>
  <xsl:copy-of select="."/>
</xsl:template>

<xsl:template match="F:component-decl/F:_obj-N_/F:s" mode="symbols-filter"/>

<xsl:template match="F:s" mode="symbols-filter">
  <xsl:element name="s" namespace="http://g95-xml.sourceforge.net/">
    <xsl:attribute name="N">
      <xsl:value-of select="@N"/>
    </xsl:attribute>
  </xsl:element>
</xsl:template>

<xsl:template match="F:derived-block">
  <xsl:element name="T-decl-stmt" namespace="http://g95-xml.sourceforge.net/">
    <xsl:variable name="name">
      <xsl:value-of select=".//F:_head_/F:derived-T-def-stmt/F:_T-N_/F:s/@N"/>
    </xsl:variable>
    <xsl:element name="kernelgen-decl-body" namespace="http://g95-xml.sourceforge.net/">
<!--
Put original statement plain code here.
-->
      <xsl:element name="c" namespace="http://g95-xml.sourceforge.net/">
        <xsl:value-of select="."/>
        <xsl:text>&#10;</xsl:text>
      </xsl:element>
<!--
From symbols take only tags, no content.
-->
      <xsl:apply-templates select=".//F:s[@N != $name]" mode="symbols-filter"/>
    </xsl:element>
<!--
In derived type definition statement insert fake symbol
to pull derived type definition into kernel routines.
-->
    <xsl:element name="entity-decl" namespace="http://g95-xml.sourceforge.net/">
      <xsl:element name="_T-N_" namespace="http://g95-xml.sourceforge.net/">
        <xsl:element name="s" namespace="http://g95-xml.sourceforge.net/">
          <xsl:attribute name="N">
            <xsl:value-of select="$name"/>
          </xsl:attribute>
        </xsl:element>
      </xsl:element>
    </xsl:element>
  </xsl:element>
  <xsl:copy-of select="."/>
</xsl:template>

<xsl:template match="node()|@*" mode="rename-obj-to-fct">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*" mode="rename-obj-to-fct"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="F:_obj-N_" mode="rename-obj-to-fct">
  <xsl:element name="_fct_inl_" namespace="http://g95-xml.sourceforge.net/">
    <xsl:copy-of select=".//F:s"/>
  </xsl:element>
</xsl:template>

<xsl:template match="F:stmt-fct-stmt" mode="process">
  <xsl:param name="name"/>
  <xsl:for-each select=".//F:_stmt-fct_/F:s">
    <xsl:if test="@N = $name">
      <xsl:text>1</xsl:text>
    </xsl:if>
  </xsl:for-each>
</xsl:template>

<xsl:template match="F:T-decl-stmt">
  <xsl:variable name="name">
    <xsl:value-of select=".//F:entity-decl/F:_obj-N_/F:s/@N"/>
  </xsl:variable>
  <xsl:variable name="is-function">
    <xsl:apply-templates select="../F:stmt-fct-stmt" mode="process">
      <xsl:with-param name="name" select="$name"/>
    </xsl:apply-templates>
  </xsl:variable>
  <xsl:choose>
<!--
If declaration is corresponding to inline function,
repalce its _obj-N_ by _fct_inl_.
-->
    <xsl:when test="$is-function != &quot;&quot;">
      <xsl:apply-templates select="." mode="rename-obj-to-fct"/>
    </xsl:when>
<!--
If declaration is not corresponding to inline function,
leave it alone.
-->
    <xsl:otherwise>
      <xsl:copy-of select="."/>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

</xsl:stylesheet>
