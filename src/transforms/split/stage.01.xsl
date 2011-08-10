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

 Purpose: Put declaration statements type part into separate tag.

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

<xsl:template match="F:entity-decl-lst" mode="process">
  <xsl:apply-templates/>
</xsl:template>

<xsl:template match="F:entity-decl-lst">
</xsl:template>

<xsl:template match="F:T-decl-stmt">
  <xsl:element name="T-decl-stmt-orig" namespace="http://g95-xml.sourceforge.net/">
    <xsl:apply-templates/>    
    <xsl:element name="entity-decl-lst" namespace="http://g95-xml.sourceforge.net/">
      <xsl:apply-templates select=".//F:entity-decl-lst" mode="process"/>
    </xsl:element>
  </xsl:element>
  <xsl:element name="T-decl-stmt" namespace="http://g95-xml.sourceforge.net/">
    <xsl:element name="gforscale-decl-body" namespace="http://g95-xml.sourceforge.net/">
      <xsl:apply-templates/>    
    </xsl:element>
    <xsl:element name="entity-decl-lst" namespace="http://g95-xml.sourceforge.net/">
      <xsl:apply-templates select=".//F:entity-decl-lst" mode="process"/>
    </xsl:element>
  </xsl:element>
</xsl:template>

</xsl:stylesheet>
