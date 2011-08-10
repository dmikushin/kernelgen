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

 Purpose: replace do groups with switch between put call <name>(<prototype>)
 and do groups. In case of kernel loop executed on device, assign loop index
 with upper boundary value after loop is complete.

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

<xsl:template match="F:do-grid"/>

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
  <xsl:element name="routine" namespace="http://g95-xml.sourceforge.net/">
    <xsl:variable name="is-module">
      <xsl:value-of select="F:stmt-lst/F:module-stmt/F:_module-N_/F:s/@N"/>
    </xsl:variable>
<!--
In case of module routine, do nothing, just digg deeper.
-->
    <xsl:choose>
      <xsl:when test="$is-module != &quot;&quot;">
        <xsl:apply-templates/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:variable name="routine-name">
          <xsl:value-of select=".//F:routine-name"/>
        </xsl:variable>
<!--
Create element with allocatable symbols module and
helper functions.
-->
        <xsl:element name="desc-helper" namespace="http://g95-xml.sourceforge.net/">
<!--
Setup use statements.
-->
          <xsl:variable name="use-stmts">
            <xsl:for-each select=".//F:modules/F:use-stmt[not(@self)]">
              <xsl:copy-of select="."/>
              <xsl:text>&#10;</xsl:text>
            </xsl:for-each>
          </xsl:variable>
<!--
Open module.
-->
          <xsl:text>module&#32;</xsl:text>
          <xsl:value-of select=".//F:routine-name"/>
          <xsl:text>_gforscale_module_uses&#10;</xsl:text>
          <xsl:copy-of select="$use-stmts"/>
          <xsl:text>end&#32;module&#32;</xsl:text>
          <xsl:value-of select=".//F:routine-name"/>
          <xsl:text>_gforscale_module_uses&#10;</xsl:text>

          <xsl:text>module&#32;</xsl:text>
          <xsl:value-of select=".//F:routine-name"/>
          <xsl:text>_gforscale_module&#10;</xsl:text>
          <xsl:text>USE&#32;GFORSCALE&#10;</xsl:text>
<!--
Add interfaces for comparison functions.
-->
          <xsl:for-each select=".//F:do-group[@routine-name = $routine-name]">
<!--
Determine routine index to make unique name.
-->
            <xsl:variable name="routine-index">
              <xsl:value-of select="count(preceding::F:do-group[@routine-name = $routine-name]) + 1"/>
            </xsl:variable>
<!--
Kernel configuration variable.
-->
            <xsl:variable name="routine-config">
              <xsl:value-of select="$routine-name"/>
              <xsl:text>_loop_</xsl:text> 
              <xsl:value-of select="$routine-index"/>
              <xsl:text>_gforscale_config</xsl:text>
            </xsl:variable>
            <xsl:text>&#10;type(gforscale_kernel_config),&#32;bind(C)&#32;::&#32;</xsl:text>
            <xsl:value-of select="$routine-config"/>
            <xsl:text>&#10;</xsl:text>
          </xsl:for-each>
<!--
Open interface section.
-->
          <xsl:text>&#10;interface&#10;</xsl:text>
<!--
Add interfaces for comparison functions.
-->
          <xsl:for-each select=".//F:do-group[@routine-name = $routine-name]">
<!--
Determine routine index to make unique name.
-->
            <xsl:variable name="routine-index">
              <xsl:value-of select="count(preceding::F:do-group[@routine-name = $routine-name]) + 1"/>
            </xsl:variable>
            <xsl:text>function&#32;</xsl:text>
            <xsl:value-of select="$routine-name"/>
            <xsl:text>_loop_</xsl:text>  
            <xsl:value-of select="$routine-index"/>
            <xsl:text>_gforscale_compare()&#10;end&#32;function&#10;</xsl:text>
          </xsl:for-each>
<!--
For each kernel routine insert an interface to
launching function. All kernel routines launch with
the same gforscale_launch_ call, but different explicit
interfaces are used to pass pointers to allocatable
arguments.
-->
          <xsl:for-each select=".//F:do-group[@routine-name = $routine-name]">
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
              <xsl:text>_gforscale_desc</xsl:text>
            </xsl:variable>
<!--
Insert routines for routine arguments.
-->
            <xsl:for-each select=".//F:args/F:s[@allocatable = 1]">
              <xsl:text>&#10;function&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>(</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>)&#10;USE&#32;</xsl:text>
              <xsl:value-of select="$routine-name"/>
              <xsl:text>_gforscale_module_uses&#10;</xsl:text>
              <!--<xsl:copy-of select="$use-stmts"/>-->
              <xsl:text>integer(8)&#32;::&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>&#10;</xsl:text>
              <xsl:variable name="symbol-name">
                <xsl:value-of select="@N"/>
              </xsl:variable>
              <xsl:choose>
                <xsl:when test="count(../../F:definitions/F:T-decl-stmt[F:entity-decl/F:_obj-N_/F:s/@N = $symbol-name]) = 1">
                  <xsl:copy-of select="../../F:definitions/F:T-decl-stmt[F:entity-decl/F:_obj-N_/F:s/@N = $symbol-name]"/>
                </xsl:when>
                <xsl:otherwise>
                  <xsl:copy-of select="../../F:definitions/F:T-decl-stmt[(@global = 0) and (F:entity-decl/F:_obj-N_/F:s/@N = $symbol-name)]"/>
                </xsl:otherwise>
              </xsl:choose>
              <xsl:text>end&#32;function&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>&#10;</xsl:text>
            </xsl:for-each>
<!--
Insert routines for modules symbols.
-->
            <xsl:for-each select=".//F:modsyms/F:s[@allocatable = 1]">
              <xsl:text>&#10;function&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>(desc)&#10;</xsl:text>
              <xsl:text>integer(8)&#32;::&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>&#10;</xsl:text>
              <xsl:value-of select="@T"/>
              <xsl:if test="@kind != &quot;&quot;">
                <xsl:text>(</xsl:text>
                <xsl:value-of select="@kind"/>
                <xsl:text>)</xsl:text>
              </xsl:if>
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
              <xsl:text>&#32;::&#32;desc&#10;</xsl:text>
              <xsl:text>end&#32;function&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>&#10;</xsl:text>
            </xsl:for-each>
          </xsl:for-each>
<!--
Close interface section and module.
-->
          <xsl:text>&#10;end&#32;interface&#10;&#10;end&#32;module&#32;</xsl:text>
          <xsl:value-of select=".//F:routine-name"/>
          <xsl:text>_gforscale_module&#10;</xsl:text>
<!--
If there are any allocatable symbols,
provide implementations of helper functions.
-->
          <xsl:for-each select=".//F:do-group[@routine-name = $routine-name]">
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
              <xsl:text>_gforscale_desc</xsl:text>
            </xsl:variable>
<!--
Insert routines for routine arguments.
-->
            <xsl:for-each select=".//F:args/F:s[@allocatable = 1]">
              <xsl:text>&#10;function&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>(</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>)&#10;</xsl:text>
              <xsl:text>integer(8)&#32;::&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>&#10;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>&#32;=&#32;loc(</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>)&#10;</xsl:text>
              <xsl:text>end&#32;function&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>&#10;</xsl:text>
            </xsl:for-each>
<!--
Insert routines for modules symbols.
-->
            <xsl:for-each select=".//F:modsyms/F:s[@allocatable = 1]">
              <xsl:text>&#10;function&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>(desc)&#10;</xsl:text>
              <xsl:text>integer(8)&#32;::&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>&#10;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>&#32;=&#32;loc(desc)&#10;</xsl:text>
              <xsl:text>end&#32;function&#32;</xsl:text>
              <xsl:value-of select="$loop-routine-name"/>
              <xsl:text>_</xsl:text>
              <xsl:value-of select="@N"/>
              <xsl:text>&#10;</xsl:text>
            </xsl:for-each>
          </xsl:for-each>
          <xsl:text>&#10;</xsl:text>
        </xsl:element>        
        <xsl:apply-templates/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:element>
</xsl:template>

<xsl:template match="F:subr-stmt|F:fct-stmt|F:program-stmt">
  <xsl:apply-templates/>
<!--
Add use statement for module with helper functions,
if there are any do-groups for the current routine.
-->
  <xsl:text>&#10;USE&#32;GFORSCALE</xsl:text>
  <xsl:variable name="routine-name">
    <xsl:value-of select=".//F:routine-name"/>
  </xsl:variable>
  <xsl:if test="count(..//F:do-group[@routine-name = $routine-name]) > 0">
    <xsl:text>&#10;USE&#32;</xsl:text>
    <xsl:value-of select="$routine-name"/>
    <xsl:text>_gforscale_module&#10;</xsl:text>
  </xsl:if>
</xsl:template>

<xsl:template match="F:do-group">
  <xsl:variable name="routine-name">
    <xsl:value-of select="@routine-name"/>
  </xsl:variable>
  <xsl:variable name="loop-index">
    <xsl:value-of select="count(preceding::F:do-group[@routine-name = $routine-name]) + 1"/>
  </xsl:variable>
  <xsl:variable name="loops-count">
    <xsl:value-of select="$loop-index + count(following::F:do-group[@routine-name = $routine-name])"/>
  </xsl:variable>
<!--
Cook routine name.
-->
  <xsl:variable name="loop-routine-name">
    <xsl:value-of select=".//@routine-name"/>
    <xsl:text>_loop_</xsl:text>
    <xsl:value-of select="$loop-index"/>
    <xsl:text>_gforscale</xsl:text>
  </xsl:variable>
<!--
Kernel configuration variable.
-->
  <xsl:variable name="routine-config">
    <xsl:value-of select=".//@routine-name"/>
    <xsl:text>_loop_</xsl:text> 
    <xsl:value-of select="$loop-index"/>
    <xsl:text>_gforscale_config</xsl:text>
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
Cook API call prototype.
-->
  <xsl:variable name="api-call">
<!--
Setup compute grid ranges.
-->
    <xsl:for-each select=".//F:do/F:do-grid[@index = &quot;0&quot;]">
      <xsl:value-of select=".//F:start"/>
      <xsl:text>,&#32;</xsl:text>
      <xsl:value-of select=".//F:end"/>
      <xsl:text>,&#32;</xsl:text>
    </xsl:for-each>
    <xsl:if test="not(.//F:do/F:do-grid[@index = &quot;0&quot;])">
      <xsl:text>0,&#32;0,&#32;</xsl:text>
    </xsl:if>
    <xsl:for-each select=".//F:do/F:do-grid[@index = &quot;1&quot;]">
      <xsl:value-of select=".//F:start"/>
      <xsl:text>,&#32;</xsl:text>
      <xsl:value-of select=".//F:end"/>
      <xsl:text>,&#32;</xsl:text>
    </xsl:for-each>
    <xsl:if test="not(.//F:do/F:do-grid[@index = &quot;1&quot;])">
      <xsl:text>0,&#32;0,&#32;</xsl:text>
    </xsl:if>
    <xsl:for-each select=".//F:do/F:do-grid[@index = &quot;2&quot;]">
      <xsl:value-of select=".//F:start"/>
      <xsl:text>,&#32;</xsl:text>
      <xsl:value-of select=".//F:end"/>
      <xsl:text>,&#32;</xsl:text>
    </xsl:for-each>
    <xsl:if test="not(.//F:do/F:do-grid[@index = &quot;2&quot;])">
      <xsl:text>0,&#32;0,&#32;</xsl:text>
    </xsl:if>
<!--
The number of arguments and used modules symbols.
-->
    <xsl:value-of select="$nargs"/>
    <xsl:text>,&#32;</xsl:text>
    <xsl:value-of select="$nmodsyms"/>
<!--
Arguments.
-->
    <xsl:for-each select=".//F:args/F:s">
      <xsl:text>,&#32;</xsl:text>
      <xsl:value-of select="."/>
      <xsl:text>,&#32;sizeof(</xsl:text>
      <xsl:value-of select="."/>
      <xsl:text>)</xsl:text>
      <xsl:text>,&#32;</xsl:text>
      <xsl:choose>
        <xsl:when test="@allocatable = 1">
          <xsl:value-of select="$loop-routine-name"/>
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
          <xsl:value-of select="$loop-routine-name"/>
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
  </xsl:variable>
<!--
Insert loop version selecting and launching blocks.
-->
  <xsl:text>&#10;!$GFORSCALE&#32;SELECT&#32;</xsl:text>
  <xsl:value-of select="$loop-routine-name"/>
  <xsl:text>&#10;if&#32;(</xsl:text>
  <xsl:value-of select="$routine-config"/>
  <xsl:text>%runmode&#32;.ne.&#32;gforscale_runmode_host)&#32;then</xsl:text>
  <xsl:text>&#10;!$GFORSCALE&#32;CALL&#32;</xsl:text>
  <xsl:value-of select="$loop-routine-name"/>
  <xsl:text>&#10;&#32;&#32;call&#32;gforscale_launch(</xsl:text>
  <xsl:value-of select="$routine-config"/>
  <xsl:text>,&#32;</xsl:text>
  <xsl:value-of select="$api-call"/>
  <xsl:text>)&#10;</xsl:text>
<!--
Assign loop index with upper boundary value after loop is complete.
-->
  <xsl:for-each select=".//F:do/F:do-grid">
    <xsl:value-of select=".//F:index"/>
    <xsl:text>&#32;=&#32;</xsl:text>
    <xsl:value-of select=".//F:end"/>
    <xsl:text>&#32;+&#32;1&#10;</xsl:text>
  </xsl:for-each>
<!--
Finish kernel loop block.
-->
  <xsl:text>!$GFORSCALE&#32;END&#32;CALL&#32;</xsl:text>
  <xsl:value-of select="$loop-routine-name"/>
  <xsl:text>&#10;endif</xsl:text>
  <xsl:text>&#10;if&#32;((iand(</xsl:text>
  <xsl:value-of select="$routine-config"/>
  <xsl:text>%runmode,&#32;gforscale_runmode_host)&#32;.eq.&#32;1)&#32;</xsl:text>
  <xsl:text>.or.&#32;(gforscale_get_last_error()&#32;.ne.&#32;0))&#32;then</xsl:text>
  <xsl:text>&#10;!$GFORSCALE&#32;LOOP&#32;</xsl:text>
  <xsl:value-of select="$loop-routine-name"/>
  <xsl:text>&#10;</xsl:text>  
  <xsl:apply-templates select="F:do"/>
  <xsl:text>&#10;!$GFORSCALE&#32;END&#32;LOOP&#32;</xsl:text>
  <xsl:value-of select="$loop-routine-name"/>
  <xsl:text>&#10;endif</xsl:text>
  <xsl:text>&#10;if&#32;((</xsl:text>
  <xsl:value-of select="$routine-config"/>
  <xsl:text>%compare&#32;.eq.&#32;1)&#32;</xsl:text>
  <xsl:text>.and.&#32;(gforscale_get_last_error()&#32;.eq.&#32;0))&#32;then</xsl:text>
  <xsl:text>&#10;&#32;&#32;call&#32;gforscale_compare(</xsl:text>
  <xsl:value-of select="$routine-config"/>
  <xsl:text>,&#32;</xsl:text>
  <xsl:value-of select="$loop-routine-name"/>
  <xsl:text>_compare,&#32;gforscale_compare_maxdiff)&#10;endif</xsl:text>
  <xsl:text>&#10;!$GFORSCALE&#32;END&#32;SELECT&#32;</xsl:text>
  <xsl:value-of select="$loop-routine-name"/>
  <xsl:text>&#10;</xsl:text>
</xsl:template>

</xsl:stylesheet>
