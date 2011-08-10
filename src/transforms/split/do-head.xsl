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

 Purpose: add do-group head tag with used symbols, their definitions, modules and other data.

-->

<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:F="http://g95-xml.sourceforge.net/"
xmlns:exsl="http://exslt.org/common" extension-element-prefixes="exsl">

<xsl:template match="node()|@*"> 
  <xsl:param name="depth"/>
  <xsl:param name="definitions"/>
  <xsl:param name="modules"/>
  <xsl:param name="implicit"/>
  <xsl:param name="interfaces"/>
  <xsl:param name="routine-name"/>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*">
      <xsl:with-param name="depth" select="$depth"/>
      <xsl:with-param name="definitions" select="$definitions"/>
      <xsl:with-param name="modules" select="$modules"/>
      <xsl:with-param name="implicit" select="$implicit"/>
      <xsl:with-param name="interfaces" select="$interfaces"/>
      <xsl:with-param name="routine-name" select="$routine-name"/>
    </xsl:apply-templates>
  </xsl:copy>
</xsl:template>
<!-- 
Filtering interfaces while selecting definitions
-->
<xsl:template match="//F:program-unit/F:*" mode="definitions-filter">
    <xsl:apply-templates select="node()|@*" mode="definitions-filter"/>
</xsl:template>

<xsl:template match="F:T-decl-stmt" mode="definitions-filter">
  <xsl:copy-of select="."/>
</xsl:template>

<xsl:template match="F:program-unit" mode="definitions-filter"/>
<xsl:template match="F:intf-block" mode="definitions-filter"/>
<xsl:template match="F:C" mode="definitions-filter"/>
<xsl:template match="F:c" mode="definitions-filter"/>
<xsl:template match="F:cont-free" mode="definitions-filter"/>
<xsl:template match="F:str" mode="definitions-filter"/>
<!-- 
Filtering interfaces while selecting used modules
-->
<xsl:template match="//F:program-unit/F:*" mode="modules-filter">
    <xsl:apply-templates select="node()|@*" mode="modules-filter"/>
</xsl:template>

<xsl:template match="F:use-stmt" mode="modules-filter">
  <xsl:copy-of select="."/>
</xsl:template>

<xsl:template match="F:program-unit" mode="modules-filter"/>
<xsl:template match="F:intf-block" mode="modules-filter"/>
<xsl:template match="F:C" mode="modules-filter"/>
<xsl:template match="F:c" mode="modules-filter"/>
<xsl:template match="F:cont-free" mode="modules-filter"/>
<xsl:template match="F:str" mode="modules-filter"/>
<!-- 
Filtering interfaces while selecting implicit none statements
-->
<xsl:template match="//F:program-unit/F:*" mode="implicit-none-filter">
    <xsl:apply-templates select="node()|@*" mode="implicit-none-filter"/>
</xsl:template>

<xsl:template match="F:implicit-none-stmt" mode="implicit-none-filter">
  <xsl:copy-of select="."/>
</xsl:template>

<xsl:template match="F:program-unit" mode="implicit-none-filter"/>
<xsl:template match="F:intf-block" mode="implicit-none-filter"/>
<xsl:template match="F:C" mode="implicit-none-filter"/>
<xsl:template match="F:c" mode="implicit-none-filter"/>
<xsl:template match="F:cont-free" mode="implicit-none-filter"/>
<xsl:template match="F:str" mode="implicit-none-filter"/>
<!-- 
Select used interfaces
-->
<xsl:template match="//F:program-unit/F:*" mode="interfaces">
    <xsl:apply-templates select="node()|@*" mode="interfaces"/>
</xsl:template>

<xsl:template match="F:intf-block" mode="interfaces">
  <xsl:copy-of select="."/>
</xsl:template>

<xsl:template match="F:C" mode="interfaces"/>
<xsl:template match="F:c" mode="interfaces"/>
<xsl:template match="F:cont-free" mode="interfaces"/>
<xsl:template match="F:str" mode="interfaces"/>

<xsl:template match="F:program-unit">
  <xsl:param name="depth"/>
  <xsl:param name="definitions"/>
  <xsl:param name="modules"/>
  <xsl:param name="implicit"/>
  <xsl:param name="interfaces"/>
  <xsl:param name="routine-name"/>
  <xsl:variable name="is-module">
    <xsl:value-of select="F:stmt-lst/F:module-stmt/F:_module-N_/F:s/@N"/>
  </xsl:variable>
<!-- 
Pass through nested nodes, selecting declaration statements,
filtering out those comming from nested interfaces.
-->
  <xsl:variable name="definitions-local">
    <xsl:apply-templates match=".//F:*" mode="definitions-filter"/>
  </xsl:variable>
<!--
Gather new list of definitions, marking global (upper) ones
and local - defined locally.
-->
  <xsl:variable name="definitions-ext">
    <xsl:for-each select="exsl:node-set($definitions)/F:T-decl-stmt">
<!--
Add only those upper level definitions that do
not conflict with local ones.
-->
      <xsl:if test="count(exsl:node-set($definitions-local)/F:T-decl-stmt[count(F:symbols/F:s[(@type = &quot;object&quot;) and (@N = $object)]) = 1]) = 0">
        <xsl:element name="T-decl-stmt" namespace="http://g95-xml.sourceforge.net/">
          <xsl:attribute name="global">
            <xsl:value-of select="1"/>
          </xsl:attribute>
          <xsl:apply-templates/>
        </xsl:element>
      </xsl:if>
    </xsl:for-each>
    <xsl:for-each select="exsl:node-set($definitions-local)/F:T-decl-stmt">
      <xsl:element name="T-decl-stmt" namespace="http://g95-xml.sourceforge.net/">
        <xsl:attribute name="global">
          <xsl:value-of select="0"/>
        </xsl:attribute>
        <xsl:apply-templates/>
      </xsl:element>
    </xsl:for-each>
  </xsl:variable>
<!-- 
Pass through nested nodes, selecting used modules,
filtering out those comming from nested interfaces.
-->
  <xsl:variable name="modules-ext">
<!--
If the current program unit is module,
add itself as used module.
-->
    <xsl:if test="$is-module != &quot;&quot;">
      <xsl:element name="use-stmt" namespace="http://g95-xml.sourceforge.net/">
        <xsl:attribute name="self">
          <xsl:value-of select="1"/>
        </xsl:attribute>
        <xsl:element name="c" namespace="http://g95-xml.sourceforge.net/">
          <xsl:text>USE&#32;</xsl:text>
        </xsl:element>
        <xsl:element name="_module-N_" namespace="http://g95-xml.sourceforge.net/">
          <xsl:element name="s" namespace="http://g95-xml.sourceforge.net/">
            <xsl:attribute name="N" select="$is-module"/>
            <xsl:element name="c" namespace="http://g95-xml.sourceforge.net/">
              <xsl:value-of select="$is-module"/>
            </xsl:element>
          </xsl:element>
        </xsl:element>
      </xsl:element>
    </xsl:if>
    <xsl:copy-of select="$modules"/>
    <xsl:apply-templates match=".//F:*" mode="modules-filter"/>
  </xsl:variable>
<!-- 
Pass through nested nodes, selecting used modules,
filtering out those comming from nested interfaces.
-->
  <xsl:variable name="implicit-ext">
    <xsl:apply-templates match=".//F:*" mode="implicit-none-filter"/>
  </xsl:variable>
<!-- 
Pass through nested nodes, selecting used interfaces.
-->
  <xsl:variable name="interfaces-ext">
    <xsl:apply-templates match=".//F:*" mode="interfaces"/>
  </xsl:variable>
  <xsl:element name="routine" namespace="http://g95-xml.sourceforge.net/">
    <xsl:text>&#10;</xsl:text>
    <xsl:apply-templates match="//F:_program-N_|//F:_subr-N_|//F:_fct-N_|//F:do-construct">
      <xsl:with-param name="depth" select="0"/>
      <xsl:with-param name="definitions">
        <xsl:copy-of select="$definitions-ext"/>
      </xsl:with-param>
      <xsl:with-param name="modules">
        <xsl:copy-of select="$modules-ext"/>
      </xsl:with-param>
      <xsl:with-param name="implicit">
        <xsl:copy-of select="$implicit-ext"/>
      </xsl:with-param>
      <xsl:with-param name="interfaces">
        <xsl:copy-of select="$interfaces-ext"/>
      </xsl:with-param>
      <xsl:with-param name="routine-name">
        <xsl:value-of select=".//F:_program-N_//F:s/@N|.//F:_subr-N_/F:s/@N|.//F:_fct-N_/F:s/@N"/>
      </xsl:with-param>
    </xsl:apply-templates>
  </xsl:element>
  <xsl:text>&#10;</xsl:text>
</xsl:template>

<xsl:template match="F:_program-N_|F:_subr-N_|F:_fct-N_">
  <xsl:element name="routine-name" namespace="http://g95-xml.sourceforge.net/">
    <xsl:value-of select=".//F:s/@N"/>
  </xsl:element>
</xsl:template>
<!--
Do not take symbols-labels as dependencies.
-->
<xsl:template match="F:_block-N_/F:s" mode="symbols-filter"/>

<xsl:template match="F:s" mode="symbols-filter">
  <xsl:copy-of select="."/>
</xsl:template>

<xsl:template match="F:do[@dim = 0]">
  <xsl:param name="depth"/>
  <xsl:param name="definitions"/>
  <xsl:param name="modules"/>
  <xsl:param name="implicit"/>
  <xsl:param name="interfaces"/>
  <xsl:param name="routine-name"/>
<!--
Create the do-group head node.
-->
  <xsl:element name="do-group" namespace="http://g95-xml.sourceforge.net/">
    <xsl:attribute name="routine-name">
      <xsl:value-of select="$routine-name"/>
    </xsl:attribute>
    <xsl:attribute name="ndims">
      <xsl:value-of select="@ndims"/>
    </xsl:attribute>
<!--
Clone context info into header.
-->
    <xsl:element name="symbols" namespace="http://g95-xml.sourceforge.net/">
      <xsl:apply-templates select=".//F:s" mode="symbols-filter"/>
    </xsl:element>        
    <xsl:element name="modules" namespace="http://g95-xml.sourceforge.net/">
      <xsl:copy-of select="$modules"/>
    </xsl:element>        
    <xsl:element name="definitions" namespace="http://g95-xml.sourceforge.net/">
      <xsl:copy-of select="$definitions"/>
    </xsl:element>
    <xsl:element name="implicit" namespace="http://g95-xml.sourceforge.net/">
      <xsl:copy-of select="$implicit"/>
    </xsl:element>
    <xsl:element name="interfaces" namespace="http://g95-xml.sourceforge.net/">
      <xsl:copy-of select="$interfaces"/>
    </xsl:element>
<!--
Copy original do tag.
-->
    <xsl:copy-of select="."/>
  </xsl:element>
</xsl:template>

</xsl:stylesheet>
