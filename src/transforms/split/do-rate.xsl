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

 Purpose: For each stack of do loops select one group with the
 maximum number of dimensions, and discard all others.
 
 Using sorting example provided by Dimitre Novatchev at
 http://www.stylusstudio.com/xsllist/200308/post30920.html

-->

<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:F="http://g95-xml.sourceforge.net/"
xmlns:exsl="http://exslt.org/common" extension-element-prefixes="exsl">

<xsl:template match="node()|@*"> 
  <xsl:param name="best-ndims"/>
  <xsl:param name="best-assigned"/>
  <xsl:param name="best-current"/>
  <xsl:param name="axises"/>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*">
      <xsl:with-param name="best-ndims" select="$best-ndims"/>
      <xsl:with-param name="best-assigned" select="$best-assigned"/>
      <xsl:with-param name="best-current" select="$best-current"/>
      <xsl:with-param name="axises" select="$axises"/>
    </xsl:apply-templates>
  </xsl:copy>
</xsl:template>

<xsl:template name="max">
  <xsl:param name="loops"/>
  <xsl:for-each select="$loops">
    <xsl:sort select="@ndims" data-type="number" order="descending"/>
    <xsl:if test="position() = 1">
      <xsl:value-of select="@ndims"/>
    </xsl:if>
  </xsl:for-each>
</xsl:template>

<xsl:template match="F:do[@dim != 0]">
  <xsl:param name="best-ndims"/>
  <xsl:param name="best-assigned"/>
  <xsl:param name="best-current"/>
  <xsl:param name="axises"/>
<!--
Copy do tag, if it belongs to the best group,
otherwise drop it.
-->
  <xsl:choose>
    <xsl:when test="$best-current = 1">
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
        <xsl:copy-of select="$axises"/>
        <xsl:apply-templates>
          <xsl:with-param name="best-ndims" select="$best-ndims"/>
          <xsl:with-param name="best-assigned" select="$best-assigned"/>
          <xsl:with-param name="best-current" select="$best-current"/>
          <xsl:with-param name="axises" select="$axises"/>
        </xsl:apply-templates>
      </xsl:element>
    </xsl:when>
    <xsl:otherwise>
      <xsl:apply-templates>
        <xsl:with-param name="best-ndims" select="$best-ndims"/>
        <xsl:with-param name="best-assigned" select="$best-assigned"/>
        <xsl:with-param name="best-current" select="$best-current"/>
      </xsl:apply-templates>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match="F:do[@dim = 0]">
  <xsl:param name="best-ndims"/>
  <xsl:param name="best-assigned"/>
  <xsl:param name="best-current"/>
<!--
Set assignment marker initial value, if not set.
-->
  <xsl:variable name="best-assigned-new">
    <xsl:choose>
      <xsl:when test="$best-assigned = &quot;&quot;">
        <xsl:value-of select="0"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$best-assigned"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:variable>
<!--
If the best ndims value is not already set,
or is not yet reached and can never be reached,
calculate it.
-->
  <xsl:variable name="best-ndims-new">
    <xsl:choose>
      <xsl:when test="($best-ndims = &quot;&quot;) or (($best-assigned-new = 0) and ($best-ndims != &quot;&quot;) and not(.//F:do[@ndims = $best-ndims]))">
        <xsl:choose>
          <xsl:when test="count(.//F:do[@dim = 0]) > 0">
            <xsl:call-template name="max">
              <xsl:with-param name="loops" select=".//F:do[@dim = 0]"/>
            </xsl:call-template>
          </xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="@ndims"/>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$best-ndims"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:variable>
<!--
Check if current loop is best.
Note the best dim lookup template above works only
for inner loops, so ">" sign in condition below
is intended to account the current loop dimension
as well.
-->
  <xsl:choose>
    <xsl:when test="($best-assigned-new = 0) and (@ndims >= $best-ndims-new)">
<!--
Copy do tag.
-->
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
Create the default layout of grid axises.
-->
        <xsl:variable name="axises">
          <xsl:element name="axises" namespace="http://g95-xml.sourceforge.net/">
            <xsl:element name="axis" namespace="http://g95-xml.sourceforge.net/">
              <xsl:attribute name="index">
                <xsl:value-of select="0"/>
              </xsl:attribute>
              <xsl:attribute name="name">
                <xsl:text>x</xsl:text>
              </xsl:attribute>
            </xsl:element>
            <xsl:element name="axis" namespace="http://g95-xml.sourceforge.net/">
              <xsl:attribute name="index">
                <xsl:value-of select="1"/>
              </xsl:attribute>
              <xsl:attribute name="name">
                <xsl:text>y</xsl:text>
              </xsl:attribute>
            </xsl:element>
            <xsl:element name="axis" namespace="http://g95-xml.sourceforge.net/">
              <xsl:attribute name="index">
                <xsl:value-of select="2"/>
              </xsl:attribute>
              <xsl:attribute name="name">
                <xsl:text>z</xsl:text>
              </xsl:attribute>
            </xsl:element>
          </xsl:element>
        </xsl:variable>
        <xsl:copy-of select="$axises"/>
<!--
Notify about the best loop selection.
-->
        <xsl:message>
          <xsl:value-of select="count(preceding::F:L)"/>
          <xsl:text>:&#32;selecting this loop&#10;</xsl:text>
        </xsl:message>
        <xsl:apply-templates>
          <xsl:with-param name="best-ndims" select="$best-ndims-new"/>
          <xsl:with-param name="best-assigned" select="1"/>
          <xsl:with-param name="best-current" select="1"/>
          <xsl:with-param name="axises" select="$axises"/>
        </xsl:apply-templates>
      </xsl:element>
    </xsl:when>
    <xsl:otherwise>
      <xsl:apply-templates>
        <xsl:with-param name="best-ndims" select="$best-ndims-new"/>
        <xsl:with-param name="best-assigned" select="$best-assigned-new"/>
        <xsl:with-param name="best-current" select="0"/>
      </xsl:apply-templates>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

</xsl:stylesheet>
