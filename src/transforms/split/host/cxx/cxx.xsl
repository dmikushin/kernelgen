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
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="F:args"/>
<xsl:template match="F:definitions"/>
<xsl:template match="F:modules"/>
<xsl:template match="F:implicit"/>
<xsl:template match="F:interfaces"/>
<xsl:template match="F:T-decl-stmt"/>
<xsl:template match="F:do-grid"/>

<xsl:template match="F:routine">
  <xsl:for-each select=".//F:do-group">
    <xsl:variable name="routine-name">
      <xsl:value-of select="@routine-name"/>
    </xsl:variable>
<!--
Determine routine index to make unique name.
-->
    <xsl:variable name="loop-index">
      <xsl:value-of select="count(preceding::F:do-group[@routine-name = $routine-name]) + 1"/>
    </xsl:variable>
<!--
Also count the total number of kernels for the current routine.
-->
    <xsl:variable name="loop-count">
      <xsl:value-of select="$loop-index + count(following::F:do-group[@routine-name = $routine-name])"/>
    </xsl:variable>
<!--
Cook kernel routine name.
-->
    <xsl:variable name="kernel-routine-name">
      <xsl:value-of select="$routine-name"/>
      <xsl:text>_loop_</xsl:text>  
      <xsl:value-of select="$loop-index"/>
      <xsl:text>_gforscale</xsl:text>
    </xsl:variable>
<!--
Cook init routine name.
-->
    <xsl:variable name="init-routine-name">
      <xsl:value-of select="$routine-name"/>
      <xsl:text>_loop_</xsl:text>  
      <xsl:value-of select="$loop-index"/>
      <xsl:text>_gforscale_init</xsl:text>
    </xsl:variable>
<!--
Cook free routine name.
-->
    <xsl:variable name="free-routine-name">
      <xsl:value-of select="$routine-name"/>
      <xsl:text>_loop_</xsl:text>  
      <xsl:value-of select="$loop-index"/>
      <xsl:text>_gforscale_free</xsl:text>
    </xsl:variable>
<!--
Cook config variable name.
-->
    <xsl:variable name="config-name">
      <xsl:value-of select="$routine-name"/>
      <xsl:text>_loop_</xsl:text>  
      <xsl:value-of select="$loop-index"/>
      <xsl:text>_gforscale_config</xsl:text>
    </xsl:variable>
<!--
Cook modules deps init routine name.
-->
    <xsl:variable name="initdeps-routine-name">
      <xsl:value-of select="$routine-name"/>
      <xsl:text>_loop_</xsl:text>  
      <xsl:value-of select="$loop-index"/>
      <xsl:text>_gforscale_init_deps</xsl:text>
    </xsl:variable>
<!--
Count the routine number of arguments.
-->
    <xsl:variable name="nargs">
      <xsl:value-of select="count(.//F:args/F:s)"/>
    </xsl:variable>
<!--
Count the routine number of used modules symbols.
-->
    <xsl:variable name="nmodsyms">
      <xsl:value-of select="count(.//F:modsyms/F:s[@T != &quot;&quot;])"/>
    </xsl:variable>
<!--
Insert routine header.
-->
    <xsl:text>&#10;!$GFORSCALE&#32;CXX&#32;HOST&#32;</xsl:text>
    <xsl:value-of select="$kernel-routine-name"/>
    <xsl:text>&#10;#include&#32;&lt;gforscale.h&gt;&#10;&#10;</xsl:text>
    <xsl:text>extern&#32;gforscale_kernel_config_t&#32;</xsl:text>
    <xsl:value-of select="$config-name"/>
    <xsl:text>;&#10;&#10;extern&#32;"C"&#32;void&#32;</xsl:text>
    <xsl:value-of select="$initdeps-routine-name"/>
    <xsl:text>(gforscale_kernel_config_t* config);&#10;&#10;</xsl:text>
    <xsl:text>__attribute__&#32;((__constructor__(102)))&#32;void&#32;</xsl:text>
    <xsl:value-of select="$init-routine-name"/>
    <xsl:text>()&#10;{&#10;gforscale_kernel_init(&amp;</xsl:text>
    <xsl:value-of select="$config-name"/>
    <xsl:text>,&#32;</xsl:text>
    <xsl:value-of select="$loop-index"/>
    <xsl:text>,&#32;</xsl:text>
    <xsl:value-of select="$loop-count"/>
    <xsl:text>,&#32;"</xsl:text>
    <xsl:value-of select="$routine-name"/>
    <xsl:text>",&#32;</xsl:text>
    <xsl:value-of select="$nargs"/>
    <xsl:text>,&#32;</xsl:text>
    <xsl:value-of select="$nmodsyms"/>
    <xsl:text>);&#10;</xsl:text>
    <xsl:value-of select="$initdeps-routine-name"/>
    <xsl:text>(&amp;</xsl:text>
    <xsl:value-of select="$config-name"/>
    <xsl:text>);&#10;}&#10;&#10;</xsl:text>
    <xsl:text>__attribute__&#32;((__destructor__(102)))&#32;void&#32;</xsl:text>
    <xsl:value-of select="$free-routine-name"/>
    <xsl:text>()&#10;{&#10;gforscale_kernel_free_deps(&amp;</xsl:text>
    <xsl:value-of select="$config-name"/>
    <xsl:text>);&#10;gforscale_kernel_free(&amp;</xsl:text>
    <xsl:value-of select="$config-name"/>
    <xsl:text>);&#10;}&#10;!$GFORSCALE&#32;END&#32;CXX&#32;HOST&#32;</xsl:text>
    <xsl:value-of select="$kernel-routine-name"/>
    <xsl:text>&#10;</xsl:text>
  </xsl:for-each>
</xsl:template>

</xsl:stylesheet>
