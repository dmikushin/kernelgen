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

 Purpose: Clone global symbols definitions down to symbols,
 for further filtering globally-defined symbols in next step.

-->

<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:F="http://g95-xml.sourceforge.net/"
xmlns:exsl="http://exslt.org/common" extension-element-prefixes="exsl">

<xsl:template match="node()|@*">
  <xsl:param name="global-defined-symbols-list"/>
  <xsl:param name="defined-symbols-list"/>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*">
      <xsl:with-param name="global-defined-symbols-list" select="$global-defined-symbols-list"/>
      <xsl:with-param name="defined-symbols-list" select="$defined-symbols-list"/>
    </xsl:apply-templates>
  </xsl:copy>
</xsl:template>
<!--
Clone G-S-lst down to F:do-group/F:symbols.
-->
<xsl:template match="F:fortran95">
  <xsl:param name="global-defined-symbols-list"/>
  <xsl:param name="defined-symbols-list"/>
  <xsl:variable name="global-defined-symbols-list-ext">
    <xsl:copy-of select="$global-defined-symbols-list"/>
    <xsl:copy-of select="exsl:node-set(.//F:G-S-lst)"/>
  </xsl:variable>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*">
      <xsl:with-param name="global-defined-symbols-list" select="$global-defined-symbols-list-ext"/>
      <xsl:with-param name="defined-symbols-list" select="$defined-symbols-list"/>
    </xsl:apply-templates>
  </xsl:copy>
</xsl:template>
<!--
Clone S-lst and F:user-T-lst down to F:do-group/F:symbols.
-->
<xsl:template match="F:routine">
  <xsl:param name="global-defined-symbols-list"/>
  <xsl:param name="defined-symbols-list"/>
  <xsl:variable name="defined-symbols-list-ext">
    <xsl:copy-of select="$defined-symbols-list"/>
    <xsl:copy-of select="exsl:node-set(.//F:S-lst)"/>
    <xsl:copy-of select="exsl:node-set(.//F:user-T-lst)"/>
  </xsl:variable>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*">
      <xsl:with-param name="global-defined-symbols-list" select="$global-defined-symbols-list"/>
      <xsl:with-param name="defined-symbols-list" select="$defined-symbols-list-ext"/>
    </xsl:apply-templates>
  </xsl:copy>
</xsl:template>

<xsl:template match="F:T-decl-stmt/F:symbols">
</xsl:template>

<xsl:template match="F:do-group/F:symbols">
  <xsl:param name="global-defined-symbols-list"/>
  <xsl:param name="defined-symbols-list"/>
  <xsl:copy-of select="$global-defined-symbols-list"/>
  <xsl:copy-of select="$defined-symbols-list"/>
  <xsl:copy-of select="."/>
</xsl:template>

</xsl:stylesheet>
