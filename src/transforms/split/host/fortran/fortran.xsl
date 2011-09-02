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

<xsl:template match="F:_init-E_" mode="filter-init-1"/>

<xsl:template match="F:_init-E_" mode="filter-init-2"/>

<xsl:template match="F:c" mode="filter-init-1">
  <xsl:if test="not(following-sibling::F:_init-E_)">
    <xsl:apply-templates/>
  </xsl:if>
</xsl:template>

<xsl:template match="F:c" mode="filter-init-2">
  <xsl:if test="not(following-sibling::F:_init-E_)">
    <xsl:apply-templates/>
  </xsl:if>
</xsl:template>

<xsl:template match="F:_obj-N_/F:s" mode="filter-init-2">
  <xsl:apply-templates/>
  <xsl:text>_kernelgen</xsl:text>
</xsl:template>

<xsl:template name="semicolon">
  <xsl:param name="ndims"/>
  <xsl:if test="$ndims > 0">
    <xsl:text>,&#32;:</xsl:text>
    <xsl:call-template name="semicolon">
      <xsl:with-param name="ndims">
        <xsl:value-of select="$ndims - 1"/>
      </xsl:with-param>
    </xsl:call-template>
  </xsl:if>
</xsl:template>

<xsl:template match="F:routine">
<!--
For each do-group.
-->
  <xsl:for-each select=".//F:do-group">
    <xsl:variable name="routine-name">
      <xsl:value-of select="@routine-name"/>
    </xsl:variable>
<!--
Setup use statements.
-->
    <xsl:variable name="use-stmts">
      <xsl:for-each select=".//F:use-stmt">
        <xsl:value-of select="."/>
        <xsl:text>&#10;</xsl:text>
      </xsl:for-each>
    </xsl:variable>
<!--
Setup implicit statements.
-->
    <xsl:variable name="implicit-none-stmts">
      <xsl:for-each select=".//F:implicit-none-stmt">
        <xsl:value-of select="."/>
        <xsl:text>&#10;</xsl:text>
      </xsl:for-each>
    </xsl:variable>
<!--
Determine routine index to make unique name.
-->
    <xsl:variable name="routine-index">
      <xsl:value-of select="count(preceding::F:do-group[@routine-name = $routine-name]) + 1"/>
    </xsl:variable>
<!--
Cook kernel routine name.
-->
    <xsl:variable name="kernel-routine-name">
      <xsl:value-of select="$routine-name"/>
      <xsl:text>_loop_</xsl:text>  
      <xsl:value-of select="$routine-index"/>
      <xsl:text>_kernelgen</xsl:text>
    </xsl:variable>
<!--
Cook modules deps init routine name.
-->
    <xsl:variable name="initdeps-routine-name">
      <xsl:value-of select="$routine-name"/>
      <xsl:text>_loop_</xsl:text>  
      <xsl:value-of select="$routine-index"/>
      <xsl:text>_kernelgen_init_deps</xsl:text>
    </xsl:variable>
<!--
Cook compare routine name.
-->
    <xsl:variable name="compare-routine-name">
      <xsl:value-of select="$routine-name"/>
      <xsl:text>_loop_</xsl:text>  
      <xsl:value-of select="$routine-index"/>
      <xsl:text>_kernelgen_compare</xsl:text>
    </xsl:variable>
<!--
Kernel configuration variable.
-->
    <xsl:variable name="routine-config">
      <xsl:value-of select=".//@routine-name"/>
      <xsl:text>_loop_</xsl:text> 
      <xsl:value-of select="$routine-index"/>
      <xsl:text>_kernelgen_config</xsl:text>
    </xsl:variable>
<!--
Open Fortran host code section.
-->
    <xsl:text>&#10;!$KERNELGEN&#32;FORTRAN&#32;HOST&#32;</xsl:text>
    <xsl:value-of select="$kernel-routine-name"/>
    <xsl:text>&#10;</xsl:text>
<!--
Insert modules deps init routine header.
-->
    <xsl:text>subroutine&#32;</xsl:text>
    <xsl:value-of select="$initdeps-routine-name"/>
    <xsl:text>(config)&#32;bind(C)&#10;USE&#32;KERNELGEN&#10;</xsl:text>
    <xsl:copy-of select="$use-stmts"/>
    <xsl:text>type(kernelgen_kernel_config),&#32;bind(C)&#32;::&#32;config</xsl:text>
    <xsl:text>&#10;call&#32;kernelgen_kernel_init_deps(config</xsl:text>
<!--
Used modules symbols.
-->
    <xsl:for-each select=".//F:modsyms/F:s[@T != &quot;&quot;]">
      <xsl:text>,&#32;</xsl:text>
      <xsl:value-of select="."/>
      <xsl:text>,&#32;sizeof(</xsl:text>
      <xsl:value-of select="."/>
      <xsl:text>)</xsl:text>
      <xsl:text>,&#32;</xsl:text>
      <xsl:choose>
        <xsl:when test="@allocatable = 1">
          <xsl:value-of select="$kernel-routine-name"/>
          <xsl:text>_desc_</xsl:text>
          <xsl:value-of select="."/>
          <xsl:text>(</xsl:text>
          <xsl:value-of select="."/>
          <xsl:text>)</xsl:text>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="."/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>
<!--
Insert modules deps init routine footer.
-->
    <xsl:text>)&#10;end&#32;subroutine&#32;</xsl:text>
    <xsl:value-of select="$initdeps-routine-name"/>
    <xsl:text>&#10;</xsl:text>
<!--
Insert comparison routine header.
-->
    <xsl:text>function&#32;</xsl:text>
    <xsl:value-of select="$compare-routine-name"/>
    <xsl:text>(maxdiff</xsl:text>
    <xsl:for-each select=".//F:args/F:s">
      <xsl:text>,&#32;</xsl:text>
      <xsl:value-of select="."/>
    </xsl:for-each>
    <xsl:for-each select=".//F:modsyms/F:s[@T != &quot;&quot;]">
      <xsl:text>,&#32;</xsl:text>
      <xsl:value-of select="."/>
      <xsl:text>_kernelgen_1</xsl:text>
    </xsl:for-each>
    <xsl:for-each select=".//F:args/F:s">
      <xsl:text>,&#32;</xsl:text>
      <xsl:value-of select="."/>
      <xsl:text>_kernelgen</xsl:text>
    </xsl:for-each>
    <xsl:for-each select=".//F:modsyms/F:s[@T != &quot;&quot;]">
      <xsl:text>,&#32;</xsl:text>
      <xsl:value-of select="."/>
      <xsl:text>_kernelgen_2</xsl:text>
    </xsl:for-each>
    <xsl:text>)</xsl:text>
    <xsl:text>&#10;USE&#32;KERNELGEN&#10;</xsl:text>
    <xsl:copy-of select="$use-stmts"/>
    <xsl:copy-of select="$implicit-none-stmts"/>
    <xsl:text>integer&#32;::&#32;</xsl:text>
    <xsl:value-of select="$compare-routine-name"/>
    <xsl:text>&#10;</xsl:text>
    <xsl:text>type(kernelgen_compare_maxdiff_t)&#32;::&#32;maxdiff&#10;</xsl:text>
    <xsl:text>real(8)&#32;::&#32;diff&#10;</xsl:text>
<!--
Add declarations. Take only those declarations, that define
symbols from routine's list of used symbols. Recognized types
are objects (variables).
-->
    <xsl:for-each select=".//F:definitions/F:T-decl-stmt">
      <xsl:variable name="defined-symbol-name">
        <xsl:value-of select=".//F:entity-decl/F:_obj-N_/F:s/@N"/>
      </xsl:variable>
<!--
Take only the definitions that are not global.
-->
      <xsl:if test="(count(../F:definitions/F:T-decl-stmt[F:entity-decl/F:_obj-N_/F:s/@N = $defined-symbol-name]) = 1) or (@global = 0)">
<!--
Filter out initialization part, if symbol is within
argument list (and therefore is not a parameter).
-->
        <xsl:choose>
          <xsl:when test="count(../../F:args/F:s[@N = $defined-symbol-name]) > 0">
            <xsl:apply-templates match="." mode="filter-init-1"/>
            <xsl:apply-templates match="." mode="filter-init-2"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:apply-templates match="."/>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:if>
    </xsl:for-each>
<!--
Add modules symbols declarations.
-->
    <xsl:for-each select=".//F:modsyms/F:s[@T != &quot;&quot;]">
      <xsl:choose>
        <xsl:when test="@derived = 1">
          <xsl:text>type(</xsl:text>
          <xsl:value-of select="@T"/>
          <xsl:text>)</xsl:text>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="@T"/>
          <xsl:if test="@kind != 0">
            <xsl:text>(</xsl:text>
            <xsl:value-of select="@kind"/>
            <xsl:text>)</xsl:text>
          </xsl:if>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:choose>
        <xsl:when test="@allocatable = 1">
          <xsl:text>,&#32;allocatable</xsl:text>
          <xsl:if test="@dimension > 0">
            <xsl:text>,&#32;dimension(:</xsl:text>
            <xsl:call-template name="semicolon">
              <xsl:with-param name="ndims">
                <xsl:value-of select="@dimension - 1"/>
              </xsl:with-param>
            </xsl:call-template>
            <xsl:text>)</xsl:text>
          </xsl:if>
        </xsl:when>
        <xsl:otherwise>
          <xsl:if test="@dimension > 0">
            <xsl:variable name="local-name">
              <xsl:value-of select="@N"/>
            </xsl:variable>
            <xsl:variable name="global-name">
              <xsl:value-of select="../../F:S-lst/F:l-G[@N = $local-name]/@G-N"/>
            </xsl:variable>
            <xsl:text>,&#32;dimension(</xsl:text>
            <xsl:for-each select="//F:fortran95/F:G-S-lst/F:G-var[@N = $global-name]/F:_array-spec_/F:array-spec/F:shape-spec-lst/F:shape-spec">
              <xsl:variable name="lower-bound">
                <xsl:value-of select=".//F:_lower-bound_/F:literal-E/@val"/>
                <xsl:value-of select=".//F:_lower-bound_/F:var-E/F:_S_/F:s/@N"/>
              </xsl:variable>
              <xsl:if test="$lower-bound = &quot;&quot;">
                <xsl:text>~ERROR~</xsl:text>
              </xsl:if>
              <xsl:value-of select="$lower-bound"/>
              <xsl:text>:</xsl:text>
              <xsl:variable name="upper-bound">
                <xsl:value-of select=".//F:_upper-bound_/F:literal-E/@val"/>
                <xsl:value-of select=".//F:_upper-bound_/F:var-E/F:_S_/F:s/@N"/>
              </xsl:variable>
              <xsl:if test="$upper-bound = &quot;&quot;">
                <xsl:text>~ERROR~</xsl:text>
              </xsl:if>
              <xsl:value-of select="$upper-bound"/>
              <xsl:if test="following-sibling::F:shape-spec">
                <xsl:text>,&#32;</xsl:text>
              </xsl:if>
            </xsl:for-each>
            <xsl:text>)</xsl:text>
          </xsl:if>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:text>&#32;::&#32;</xsl:text>
      <xsl:value-of select="@N"/>
      <xsl:text>_kernelgen_1,&#32;</xsl:text>
      <xsl:value-of select="@N"/>
      <xsl:text>_kernelgen_2&#10;</xsl:text>
    </xsl:for-each>
    <xsl:text>&#10;</xsl:text>
    <xsl:value-of select="$compare-routine-name"/>
    <xsl:text>&#32;=&#32;0&#10;&#10;</xsl:text>
<!--
Compare each pair of routine arguments.
-->
    <xsl:for-each select=".//F:args/F:s">
      <xsl:variable name="symbol-name">
        <xsl:value-of select="@N"/>
      </xsl:variable>
      <xsl:variable name="is-array">
        <xsl:for-each select="../../F:definitions/F:T-decl-stmt[F:entity-decl/F:_obj-N_/F:s/@N = $symbol-name]">
          <xsl:for-each select=".//F:_array-spec_">
            <xsl:value-of select="1"/>
          </xsl:for-each>
        </xsl:for-each>
      </xsl:variable>
      <xsl:variable name="max">
        <xsl:if test="$is-array != &quot;&quot;">
          <xsl:text>maxval</xsl:text>
        </xsl:if>
      </xsl:variable>
      <xsl:variable name="min">
        <xsl:if test="$is-array != &quot;&quot;">
          <xsl:text>minval</xsl:text>
        </xsl:if>
      </xsl:variable>
      <xsl:if test="@allocatable = 1">
        <xsl:text>if&#32;(allocated(</xsl:text>
        <xsl:value-of select="@N"/>
        <xsl:text>)&#32;.and.&#32;allocated(</xsl:text>        
        <xsl:value-of select="@N"/>
        <xsl:text>_kernelgen))&#32;then&#10;</xsl:text>
      </xsl:if>
      <xsl:variable name="type">
        <xsl:value-of select="../../F:definitions/F:T-decl-stmt[F:entity-decl/F:_obj-N_/F:s/@N = $symbol-name]/F:kernelgen-decl-body/F:_T-spec_/F:I-T-spec/F:_I-T-N_/F:I-T-N/@N"/>
      </xsl:variable>
      <xsl:choose>
        <xsl:when test="$type = &quot;integer&quot;">
          <xsl:text>if&#32;((</xsl:text>
          <xsl:value-of select="$max"/>
          <xsl:text>(</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>&#32;-&#32;</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>_kernelgen))&#32;.ne.&#32;0&#32;.or.&#32;(</xsl:text>
          <xsl:value-of select="$min"/>
          <xsl:text>(</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>&#32;-&#32;</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>_kernelgen))&#32;.ne.&#32;0)&#32;&amp;&#10;&#32;&#32;</xsl:text>
          <xsl:value-of select="$compare-routine-name"/>
          <xsl:text>&#32;=&#32;</xsl:text>
          <xsl:value-of select="$compare-routine-name"/>
          <xsl:text>&#32;+&#32;1&#10;</xsl:text>
        </xsl:when>
        <xsl:when test="$type = &quot;real&quot;">
          <xsl:text>diff&#32;=&#32;sqrt(</xsl:text>
          <xsl:if test="$is-array != &quot;&quot;">
            <xsl:text>sum</xsl:text>
          </xsl:if>
          <xsl:text>((</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>_kernelgen&#32;/&#32;</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>&#32;-&#32;1.0)**2)</xsl:text>
          <xsl:if test="$is-array != &quot;&quot;">
            <xsl:text>&#32;/&#32;size(</xsl:text>
            <xsl:value-of select="@N"/>
            <xsl:text>)</xsl:text>
          </xsl:if>
          <xsl:text>)</xsl:text>
          <xsl:text>&#10;print&#32;*,&#32;&quot;diff(</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>)&#32;=&#32;&quot;,&#32;diff&#10;</xsl:text>
          <xsl:text>if&#32;((diff&#32;.ne.&#32;diff)&#32;.or.&#32;</xsl:text>
          <xsl:text>(diff&#32;.ge.&#32;maxdiff%single))&#32;&amp;&#10;&#32;&#32;</xsl:text>
          <xsl:value-of select="$compare-routine-name"/>
          <xsl:text>&#32;=&#32;</xsl:text>
          <xsl:value-of select="$compare-routine-name"/>
          <xsl:text>&#32;+&#32;1&#10;</xsl:text>
        </xsl:when>
        <xsl:otherwise>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:if test="@allocatable = 1">
        <xsl:text>endif&#10;</xsl:text>
      </xsl:if>
    </xsl:for-each>
<!--
Compare each pair of routine data dependencies.
-->
    <xsl:for-each select=".//F:modsyms/F:s[@T != &quot;&quot;]">
      <xsl:variable name="max">
        <xsl:if test="@dimension > 0">
          <xsl:text>maxval</xsl:text>
        </xsl:if>
      </xsl:variable>
      <xsl:variable name="min">
        <xsl:if test="@dimension > 0">
          <xsl:text>minval</xsl:text>
        </xsl:if>
      </xsl:variable>
      <xsl:if test="@allocatable = 1">
        <xsl:text>if&#32;(allocated(</xsl:text>
        <xsl:value-of select="@N"/>
        <xsl:text>_kernelgen_1)&#32;.and.&#32;allocated(</xsl:text>        
        <xsl:value-of select="@N"/>
        <xsl:text>_kernelgen_2))&#32;then&#10;</xsl:text>
      </xsl:if>
      <xsl:choose>
        <xsl:when test="@T = &quot;integer&quot;">
          <xsl:text>if&#32;((</xsl:text>
          <xsl:value-of select="$max"/>
          <xsl:text>(</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>_kernelgen_1&#32;-&#32;</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>_kernelgen_2))&#32;.ne.&#32;0&#32;.or.&#32;(</xsl:text>
          <xsl:value-of select="$min"/>
          <xsl:text>(</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>_kernelgen_1&#32;-&#32;</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>_kernelgen_2))&#32;.ne.&#32;0)&#32;&amp;&#10;&#32;&#32;</xsl:text>
          <xsl:value-of select="$compare-routine-name"/>
          <xsl:text>&#32;=&#32;</xsl:text>
          <xsl:value-of select="$compare-routine-name"/>
          <xsl:text>&#32;+&#32;1&#10;</xsl:text>
        </xsl:when>
        <xsl:when test="@T = &quot;real&quot;">
          <xsl:text>diff = sqrt(</xsl:text>
          <xsl:if test="@dimension > 0">
            <xsl:text>sum</xsl:text>
          </xsl:if>
          <xsl:text>((</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>_kernelgen_2&#32;/&#32;</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>_kernelgen_1&#32;-&#32;1.0)**2)</xsl:text>
          <xsl:if test="@dimension > 0">
            <xsl:text>&#32;/&#32;size(</xsl:text>
            <xsl:value-of select="@N"/>
            <xsl:text>_kernelgen_1)</xsl:text>
          </xsl:if>
          <xsl:text>)</xsl:text>
          <xsl:text>&#10;print&#32;*,&#32;&quot;diff(</xsl:text>
          <xsl:value-of select="@N"/>
          <xsl:text>)&#32;=&#32;&quot;,&#32;diff&#10;</xsl:text>
          <xsl:text>if&#32;((diff&#32;.ne.&#32;diff)&#32;.or.&#32;</xsl:text>
          <xsl:text>(diff&#32;.ge.&#32;maxdiff%single))&#32;&amp;&#10;&#32;&#32;</xsl:text>
          <xsl:value-of select="$compare-routine-name"/>
          <xsl:text>&#32;=&#32;</xsl:text>
          <xsl:value-of select="$compare-routine-name"/>
          <xsl:text>&#32;+&#32;1&#10;</xsl:text>
        </xsl:when>
        <xsl:otherwise>
<!--
NOTE: comparison of symbols with other builtin or derived types is disabled
(that's a very complex task!).
-->
        </xsl:otherwise>
      </xsl:choose>
      <xsl:if test="@allocatable = 1">
        <xsl:text>endif&#10;</xsl:text>
      </xsl:if>
    </xsl:for-each>
<!--
Insert comparison routine footer.
-->
    <xsl:text>&#10;end&#32;function&#32;</xsl:text>
    <xsl:value-of select="$compare-routine-name"/>
    <xsl:text>&#10;</xsl:text>
<!--
Close Fortran host code section.
-->
    <xsl:text>!$KERNELGEN&#32;END&#32;FORTRAN&#32;HOST&#32;</xsl:text>
    <xsl:value-of select="$kernel-routine-name"/>
    <xsl:text>&#10;</xsl:text>
  </xsl:for-each>
</xsl:template>

</xsl:stylesheet>
