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

 Purpose: Exclude duplicates from used declarations list

-->

<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:F="http://g95-xml.sourceforge.net/">

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/> 
  </xsl:copy>
</xsl:template> 
 
<xsl:key name="definitions" match="F:T-decl-stmt" use="concat(generate-id(parent::F:definitions[1]),F:entity-decl/F:_obj-N_/F:s,F:entity-decl/F:_fct_inl_/F:s,F:gforscale-decl-body)"/>
 
<xsl:template match="F:definitions">
  <xsl:element name="definitions" namespace="http://g95-xml.sourceforge.net/">
    <xsl:for-each select=".//F:T-decl-stmt">
      <xsl:variable name="mykey" select="concat(generate-id(parent::F:definitions[1]),F:entity-decl/F:_obj-N_/F:s,F:entity-decl/F:_fct_inl_/F:s,F:gforscale-decl-body)"/>
      <xsl:if test="generate-id(key('definitions',$mykey)[1])=generate-id()">
        <xsl:copy-of select="."/>
      </xsl:if>
    </xsl:for-each>
  </xsl:element>
</xsl:template>
 
</xsl:stylesheet>
