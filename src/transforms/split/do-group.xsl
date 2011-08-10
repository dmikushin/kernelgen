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

 Purpose: Mark stacks of do-constructs (do-groups)
 for kernel loops.

-->

<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:F="http://g95-xml.sourceforge.net/"
xmlns:exsl="http://exslt.org/common" extension-element-prefixes="exsl">

<xsl:template match="node()|@*"> 
  <xsl:param name="depth"/>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*">
      <xsl:with-param name="depth" select="$depth"/>
    </xsl:apply-templates>
  </xsl:copy>
</xsl:template>

<xsl:template match="F:do-construct[F:_do-block_/F:block/F:_do-head_/F:do-stmt/F:_iterator_]">
  <xsl:param name="depth"/>
<!--
Set initial depth.
-->
  <xsl:variable name="depth-new">
    <xsl:choose>
      <xsl:when test="$depth != &quot;&quot;">
        <xsl:value-of select="$depth"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="0"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:variable>
<!--
Digg down to the most nested loop.
-->
  <xsl:variable name="nested">
<!--
If loop contains nested loops, go deeper.
-->
    <xsl:apply-templates match="//F:do-construct">
      <xsl:with-param name="depth" select="$depth-new + 1"/>
    </xsl:apply-templates>
  </xsl:variable>
<!--
There are two types of non-portable statements.
Statements of the "local" type have local effect (EXIT, CYCLE, etc.),
making only entire loop non-portable. In this case entire loop must be
discarded. Usable exclusion criteria is cooked during next few steps.
Statements of the "global" type affect the whole loops stack
(WRITE, PRINT, etc.). In this case entire and outer loops can be
skipped now.
-->
  <xsl:variable name="globally-nonportable-symbols" select="count(.//F:print-stmt) + count(.//F:write-stmt)"/>
  <xsl:variable name="locally-nonportable-symbols" select="count(.//F:exit-stmt) + count(.//F:cycle-stmt)"/>  
<!--
Depending on conditions, choose how to markup the
collected stack of nested loops.
-->
  <xsl:element name="do" namespace="http://g95-xml.sourceforge.net/">
    <xsl:attribute name="depth">
      <xsl:value-of select="$depth-new"/>
    </xsl:attribute>
    <xsl:attribute name="globally-nonportable-symbols">
      <xsl:value-of select="$globally-nonportable-symbols"/>
    </xsl:attribute>
    <xsl:attribute name="locally-nonportable-symbols">
      <xsl:value-of select="$locally-nonportable-symbols"/>
    </xsl:attribute>
    <xsl:choose>
<!--
If entire do-construct is alone in the statement list,
then it fits as kernel loop, so put it into open do-tag.
-->
      <xsl:when test="(count(preceding-sibling::F:*) = 0) and (count(following-sibling::F:*) = 0) and name(../../../..) = &quot;do-construct&quot;">
        <xsl:attribute name="top">
          <xsl:value-of select="0"/>
        </xsl:attribute>
      </xsl:when>
<!--
Otherwise, do-construct is in the context of other statements.
This could happen either in nested loop (that breaks necessary condition),
or when the most outer loop is reached. In both cases enclose
collected loops stack into finalizing do-tag.
-->
      <xsl:otherwise>
        <xsl:attribute name="top">
          <xsl:value-of select="1"/>
        </xsl:attribute>
      </xsl:otherwise>
    </xsl:choose>
    <xsl:copy-of select="$nested"/>
  </xsl:element>
</xsl:template>

</xsl:stylesheet>
