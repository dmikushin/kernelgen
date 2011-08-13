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

 Purpose: For each do-group remove parts containing
 non-portable symbols.

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

<xsl:template match="F:do">
  <xsl:variable name="depth" select="@depth"/>
  <xsl:variable name="id" select="@id"/>
  <xsl:variable name="nonportable-symbols">
    <xsl:choose>
      <xsl:when test=".//F:do[@depth = $depth + 1]">
        <xsl:value-of select="@nonportable-symbols - .//F:do[@depth = $depth + 1]/@nonportable-symbols"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="@nonportable-symbols"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:variable>
<!--
Create element depending on non-portable statements presence.
-->
  <xsl:choose>
    <xsl:when test="$nonportable-symbols = 0">
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
        <xsl:apply-templates/>
      </xsl:element>
    </xsl:when>
    <xsl:otherwise>
      <xsl:if test=".//F:exit-stmt">
        <xsl:message>
          <xsl:value-of select="count(preceding::F:L)"/>
          <xsl:text>:&#32;locally&#32;nonportable&#32;symbol&#32;EXIT&#10;</xsl:text>
        </xsl:message>
      </xsl:if>
      <xsl:if test=".//F:cycle-stmt">
        <xsl:message>
          <xsl:value-of select="count(preceding::F:L)"/>
          <xsl:text>:&#32;locally&#32;nonportable&#32;symbol&#32;CYCLE&#10;</xsl:text>
        </xsl:message>
      </xsl:if>
      <xsl:apply-templates/>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

</xsl:stylesheet>
