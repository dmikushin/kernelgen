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

 Purpose: assign compute grid axis to each portable do loop.

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

<xsl:template match="F:axises"/>

<xsl:template match="F:do">
  <xsl:element name="do" namespace="http://g95-xml.sourceforge.net/">
    <xsl:attribute name="id">
      <xsl:value-of select="@id"/>
    </xsl:attribute>
    <xsl:attribute name="depth">
      <xsl:value-of select="@depth"/>
    </xsl:attribute>
    <xsl:attribute name="dim">
      <xsl:value-of select="@dim"/>
    </xsl:attribute>
    <xsl:attribute name="ndims">
      <xsl:value-of select="@ndims"/>
    </xsl:attribute>
<!--
Do not add anything, if this loop has greater dimension
index, than the number of available compute grid axises.
-->
      <xsl:if test="count(./F:axises/F:axis) > @dim">
<!--
Extract loop index and ranges.
-->
        <xsl:variable name="dim">
          <xsl:value-of select="@dim"/>
        </xsl:variable>
        <xsl:variable name="ndims">
          <xsl:value-of select="@ndims"/>
        </xsl:variable>
        <xsl:variable name="name">
          <xsl:value-of select=".//F:axises/F:axis[@index = $ndims - $dim - 1]/@name"/>
        </xsl:variable>
        <xsl:element name="do-grid" namespace="http://g95-xml.sourceforge.net/">
          <xsl:attribute name="name">
            <xsl:value-of select="$name"/>
          </xsl:attribute>
          <xsl:attribute name="index">
            <xsl:value-of select="$ndims - $dim - 1"/>
          </xsl:attribute>
          <xsl:element name="index" namespace="http://g95-xml.sourceforge.net/">
            <xsl:copy-of select="./F:_do-block_/F:block/F:_do-head_/F:do-stmt/F:_iterator_/F:iterator/F:_var_"/>
          </xsl:element>
          <xsl:element name="start" namespace="http://g95-xml.sourceforge.net/">
            <xsl:copy-of select="./F:_do-block_/F:block/F:_do-head_/F:do-stmt/F:_iterator_/F:iterator/F:_start_"/>
          </xsl:element>
          <xsl:element name="end" namespace="http://g95-xml.sourceforge.net/">
            <xsl:copy-of select="./F:_do-block_/F:block/F:_do-head_/F:do-stmt/F:_iterator_/F:iterator/F:_end_"/>
          </xsl:element>
<!--
Add grid axis.
-->
          <xsl:element name="axis" namespace="http://g95-xml.sourceforge.net/">
            <xsl:value-of select="$name"/>
          </xsl:element>
        </xsl:element>
      </xsl:if>
    <xsl:apply-templates/>
  </xsl:element>
</xsl:template>

</xsl:stylesheet>
