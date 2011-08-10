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

 Purpose: For each do-construct create an argument list and list of used modules
 symbols. Do not include definitions with "parameter" attribute to lists.
 In each declaration statement move the list of used symbols to declarations.
 Report defined modules.

-->

<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:F="http://g95-xml.sourceforge.net/">

<xsl:template match="node()|@*">
  <xsl:param name="symbols"/>
  <xsl:param name="dim-x"/>
  <xsl:param name="dim-y"/>
  <xsl:param name="dim-z"/>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"> 
      <xsl:with-param name="symbols" select="$symbols"/>
      <xsl:with-param name="dim-x" select="$dim-x"/>
      <xsl:with-param name="dim-y" select="$dim-y"/>
      <xsl:with-param name="dim-z" select="$dim-z"/>
    </xsl:apply-templates>
  </xsl:copy>
</xsl:template> 

<xsl:template match="F:do-group">
  <xsl:variable name="symbols">
    <xsl:copy-of select=".//F:symbols"/>
  </xsl:variable>
  <xsl:variable name="dim-x">
    <xsl:value-of select=".//F:do-grid[@index = 0]/F:index"/>
  </xsl:variable>
  <xsl:variable name="dim-y">
    <xsl:value-of select=".//F:do-grid[@index = 1]/F:index"/>
  </xsl:variable>
  <xsl:variable name="dim-z">
    <xsl:value-of select=".//F:do-grid[@index = 2]/F:index"/>
  </xsl:variable>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"> 
      <xsl:with-param name="symbols" select="$symbols"/>
      <xsl:with-param name="dim-x" select="$dim-x"/>
      <xsl:with-param name="dim-y" select="$dim-y"/>
      <xsl:with-param name="dim-z" select="$dim-z"/>
    </xsl:apply-templates>
  </xsl:copy>
</xsl:template> 

<xsl:template match="F:T-decl-stmt">
  <xsl:param name="symbols"/>
  <xsl:param name="dim-x"/>
  <xsl:param name="dim-y"/>
  <xsl:param name="dim-z"/>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"> 
      <xsl:with-param name="symbols" select="$symbols"/>
      <xsl:with-param name="dim-x" select="$dim-x"/>
      <xsl:with-param name="dim-y" select="$dim-y"/>
      <xsl:with-param name="dim-z" select="$dim-z"/>
    </xsl:apply-templates>
    <xsl:copy-of select="$symbols"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="F:gforscale-decl-body/F:c[following-sibling::F:attr-spec-lst]">
  <xsl:if test="following-sibling::F:attr-spec-lst/F:attr-spec[(@N != &quot;private&quot;) and (@N != &quot;optional&quot;) and (@N != &quot;save&quot;) and (@N != &quot;intent-in&quot;) and (@N != &quot;intent-out&quot;) and (@N != &quot;intent-inout&quot;)]">
<!--
Rewrite text node if it is not one of:
private, optional, intent, save.
-->
    <xsl:element name="c" namespace="http://g95-xml.sourceforge.net/">
      <xsl:copy-of select="."/>
    </xsl:element>
  </xsl:if>
</xsl:template>

<xsl:template match="F:gforscale-decl-body/F:attr-spec-lst/F:c[following-sibling::F:attr-spec]">
  <xsl:if test="following-sibling::F:attr-spec[(@N != &quot;private&quot;) and (@N != &quot;optional&quot;) and (@N != &quot;save&quot;) and (@N != &quot;intent-in&quot;) and (@N != &quot;intent-out&quot;) and (@N != &quot;intent-inout&quot;)]">
<!--
Rewrite text node if it is not one of:
private, optional, intent, save.
-->
    <xsl:element name="c" namespace="http://g95-xml.sourceforge.net/">
      <xsl:copy-of select="."/>
    </xsl:element>
  </xsl:if>
</xsl:template>

<xsl:template match="F:gforscale-decl-body/F:attr-spec-lst/F:attr-spec">
  <xsl:if test="(@N != &quot;private&quot;) and (@N != &quot;optional&quot;) and (@N != &quot;save&quot;) and (@N != &quot;intent-in&quot;) and (@N != &quot;intent-out&quot;) and (@N != &quot;intent-inout&quot;)">
<!--
Rewrite attribute if it is not one of:
private, optional, intent, save.
-->
    <xsl:element name="attr-spec" namespace="http://g95-xml.sourceforge.net/">
      <xsl:attribute name="N">
        <xsl:value-of select="@N"/>
      </xsl:attribute>
      <xsl:apply-templates/>
    </xsl:element>
  </xsl:if>
</xsl:template>

<xsl:template match="F:symbols">
  <xsl:param name="symbols"/>
  <xsl:param name="dim-x"/>
  <xsl:param name="dim-y"/>
  <xsl:param name="dim-z"/>
<!--
Create arguments list.
-->
  <xsl:element name="args" namespace="http://g95-xml.sourceforge.net/">
    <xsl:for-each select=".//F:s[@global = 0]">
      <xsl:variable name="name">
        <xsl:value-of select="@N"/>
      </xsl:variable>
      <xsl:choose>
<!--
Take only those symbols, that are not indentified
as functions or subroutines and do not have
parameter attribute.
-->
        <xsl:when test="(@type = &quot;function&quot;) or (@type = &quot;subroutine&quot;) or (@type = &quot;parameter&quot;) or (@type = &quot;type&quot;)"/>
<!--
Take only those symbols, that are not indexes
of compute grid loops.
-->
        <xsl:when test="(@N = $dim-x) or (@N = $dim-y) or (@N = $dim-z)"/>
<!--
Take only those symbols, that are not indexes
in nested loops.
-->
        <xsl:when test="count(../..//F:_do-block_/F:block/F:_do-head_/F:do-stmt/F:_iterator_/F:iterator/F:_var_[. = $name]) > 0"/>
<!--
Take only those symbols, that are not used
in assignments left side as scalars.
-->
        <xsl:when test="count(../..//F:assgt-stmt/F:_assgt_[preceding-sibling::F:_var_[(count(F:var-E/F:ref-lst/F:array-element-ref) + count(F:var-E/F:ref-lst/F:array-SC-ref) = 0) and (F:var-E/F:_S_/F:s/@N = $name)]]) > 0"/>
    
        <xsl:otherwise>
          <xsl:element name="s" namespace="http://g95-xml.sourceforge.net/">
            <xsl:attribute name="N">
              <xsl:value-of select="@N"/>
            </xsl:attribute>
            <xsl:attribute name="allocatable">
              <xsl:value-of select="@allocatable"/>
            </xsl:attribute>
            <xsl:element name="c" namespace="http://g95-xml.sourceforge.net/">
              <xsl:value-of select="@N"/>
            </xsl:element>
          </xsl:element>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>  
  </xsl:element>
<!--
Create list of used modules symbols.
-->
  <xsl:element name="modsyms" namespace="http://g95-xml.sourceforge.net/">
    <xsl:for-each select=".//F:s[@global = 1]">
      <xsl:choose>
<!--
Take only those symbols, that are not indentified
as functions or subroutines and do not have
parameter attribute.
-->
        <xsl:when test="(@type = &quot;function&quot;) or (@type = &quot;subroutine&quot;) or (@type = &quot;parameter&quot;) or (@type = &quot;type&quot;)"/>
<!--
Take only those symbols, that are not indexes
of compute grid loops.
-->
        <xsl:when test="(@N = $dim-x) or (@N = $dim-y) or (@N = $dim-z)"/>
    
        <xsl:otherwise>
          <xsl:element name="s" namespace="http://g95-xml.sourceforge.net/">
            <xsl:attribute name="N">
              <xsl:value-of select="@N"/>
            </xsl:attribute>
            <xsl:attribute name="allocatable">
              <xsl:value-of select="@allocatable"/>
            </xsl:attribute>
            <xsl:attribute name="T">
              <xsl:value-of select="@T"/>
            </xsl:attribute>
            <xsl:attribute name="kind">
              <xsl:value-of select="@kind"/>
            </xsl:attribute>
            <xsl:attribute name="dimension">
              <xsl:value-of select="@dimension"/>
            </xsl:attribute>
            <xsl:attribute name="derived">
              <xsl:value-of select="@derived"/>
            </xsl:attribute>
            <xsl:element name="c" namespace="http://g95-xml.sourceforge.net/">
              <xsl:value-of select="@N"/>
            </xsl:element>
          </xsl:element>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>  
  </xsl:element>
  <xsl:copy-of select="."/>
</xsl:template>
<!--
Report defined modules names.
-->
<xsl:template match="F:module-stmt/F:_module-N_/F:s">
  <xsl:message>
    <xsl:message>
      <xsl:value-of select="count(preceding::F:L)"/>
      <xsl:text>:&#32;defines&#32;module&#32;"</xsl:text>
      <xsl:value-of select="@N"/>
      <xsl:text>"&#10;</xsl:text>
    </xsl:message>
  </xsl:message>
  <xsl:copy-of select="."/>
</xsl:template>

</xsl:stylesheet>
