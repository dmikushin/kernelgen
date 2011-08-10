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

 Purpose: Extract loops routines, include only declarations
 for symbols used in entire routine. Apply default compute grid
 setup: instead of two most outer loops, assign their indexes with
 compute grid block indexes.

-->

<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:F="http://g95-xml.sourceforge.net/">

<xsl:template match="node()|@*">
  <xsl:param name="routine-name"/>
  <xsl:param name="routine-index"/>
  <xsl:param name="grid"/>
  <xsl:param name="axis"/>
  <xsl:param name="index"/>
  <xsl:param name="start"/>
  <xsl:param name="end"/>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"> 
      <xsl:with-param name="routine-name" select="$routine-name"/>
      <xsl:with-param name="routine-index" select="$routine-index"/>
      <xsl:with-param name="grid" select="$grid"/>
      <xsl:with-param name="axis" select="$axis"/>
      <xsl:with-param name="index" select="$index"/>
      <xsl:with-param name="start" select="$start"/>
      <xsl:with-param name="end" select="$end"/>
    </xsl:apply-templates>
  </xsl:copy>
</xsl:template>

<xsl:template match="F:args"/>
<xsl:template match="F:definitions"/>
<xsl:template match="F:modules"/>
<xsl:template match="F:implicit"/>
<xsl:template match="F:interfaces"/>
<xsl:template match="F:T-decl-stmt"/>
<xsl:template match="F:do-grid"/>

<xsl:template match="F:_init-E_" mode="filter-init"/>

<xsl:template match="F:c" mode="filter-init">
  <xsl:if test="not(following-sibling::F:_init-E_)">
    <xsl:apply-templates/>
  </xsl:if>
</xsl:template>

<xsl:template match="F:routine">
  <xsl:for-each select=".//F:do-group">
    <xsl:variable name="routine-name">
      <xsl:value-of select="@routine-name"/>
    </xsl:variable>
<!--
Determine routine index to make unique name.
-->
    <xsl:variable name="routine-index">
      <xsl:value-of select="count(preceding::F:do-group[@routine-name = $routine-name]) + 1"/>
    </xsl:variable>
<!--
Cook routine name.
-->
    <xsl:variable name="loop-routine-name">
      <xsl:value-of select="$routine-name"/>
      <xsl:text>_loop_</xsl:text>  
      <xsl:value-of select="$routine-index"/>
      <xsl:text>_gforscale</xsl:text>
    </xsl:variable>
<!--
Insert routine header.
-->
    <xsl:text>&#10;!$GFORSCALE&#32;CPU&#32;DEVICE&#32;</xsl:text>
    <xsl:value-of select="$loop-routine-name"/>
    <xsl:text>&#10;subroutine&#32;</xsl:text>
    <xsl:value-of select="$loop-routine-name"/>
    <xsl:text>(</xsl:text>
    <xsl:for-each select=".//F:args/F:s">
      <xsl:if test="preceding-sibling::F:s">
        <xsl:text>,&#32;</xsl:text>
      </xsl:if>
      <xsl:value-of select="."/>
    </xsl:for-each>
    <xsl:text>)</xsl:text>
    <xsl:text>&#10;</xsl:text>
    <xsl:for-each select=".//F:use-stmt">
      <xsl:value-of select="."/>
      <xsl:text>&#10;</xsl:text>
    </xsl:for-each>
    <xsl:for-each select=".//F:implicit-none-stmt">
      <xsl:value-of select="."/>
      <xsl:text>&#10;</xsl:text>
    </xsl:for-each>
    <xsl:for-each select=".//F:intf-block">
      <xsl:value-of select="."/>
      <xsl:text>&#10;</xsl:text>
    </xsl:for-each>
<!--
Add declarations. Take only those declarations, that define
symbols from routine's list of used symbols. Recognized types
are objects (variables) and inline functions.
-->
    <xsl:for-each select=".//F:definitions/F:T-decl-stmt">
      <xsl:variable name="defined-symbol-name">
        <xsl:value-of select=".//F:entity-decl/F:_obj-N_/F:s/@N"/>
        <xsl:value-of select=".//F:entity-decl/F:_fct_inl_/F:s/@N"/>
      </xsl:variable>
      <xsl:if test="(count(../F:definitions/F:T-decl-stmt[F:entity-decl/F:_obj-N_/F:s/@N = $defined-symbol-name]) = 1) or (@global = 0)">
<!--
Filter out initialization part, if symbol is within
argument list (and therefore is not a parameter).
-->
        <xsl:choose>
          <xsl:when test="count(../../F:args/F:s[@N = $defined-symbol-name]) > 0">
            <xsl:apply-templates match="." mode="filter-init"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:apply-templates match="."/>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:if>
    </xsl:for-each>
<!--
Insert do loops bodies.
-->
    <xsl:apply-templates select="F:do"/>
<!--
Insert routine footer.
-->
    <xsl:text>&#10;end&#32;subroutine&#32;</xsl:text>
    <xsl:value-of select="$loop-routine-name"/>
    <xsl:text>&#10;!$GFORSCALE&#32;END&#32;CPU&#32;DEVICE&#32;</xsl:text>
    <xsl:value-of select="$loop-routine-name"/>
    <xsl:text>&#10;</xsl:text>
  </xsl:for-each>
</xsl:template>

</xsl:stylesheet>
