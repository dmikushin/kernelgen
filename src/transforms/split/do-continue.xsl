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

 Purpose: Replace old-style do .. continue with do .. enddo

-->

<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:F="http://g95-xml.sourceforge.net/"
xmlns:exsl="http://exslt.org/common" extension-element-prefixes="exsl">

<xsl:template match="node()|@*"> 
  <xsl:param name="depth"/>
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="F:continue-stmt[F:_stmt-label_/F:label-ref/@label]">
<!--
Remove continue statement that refers to the same label
as at least one of do-contruct headers.
-->
  <xsl:variable name="label">
    <xsl:value-of select="F:_stmt-label_/F:label-ref/@label"/>
  </xsl:variable>
  <xsl:if test="count(../F:do-construct[F:_do-block_/F:block/F:_do-head_/F:do-stmt/F:_label_/F:label-ref/@label = $label]) = 0">
    <xsl:copy-of select="."/>
  </xsl:if>
</xsl:template>

<!--
Remove labels from all numeric do-loops headers.
-->
<xsl:template match="F:do-stmt[F:_iterator_]/F:_label_"/>

<xsl:template match="F:do-construct[(F:_do-block_/F:block/F:_do-head_/F:do-stmt/F:_iterator_) and (F:_do-block_/F:block/F:_do-head_/F:do-stmt/F:_label_) and not(F:_end-do-stmt_/F:end-do-stmt)]">
<!--
If numeric do-loop does not have enddo statement, add one.
-->
   <xsl:element name="do-construct" namespace="http://g95-xml.sourceforge.net/">
     <xsl:apply-templates/>
     <xsl:element name="_end-do-stmt_" namespace="http://g95-xml.sourceforge.net/">
       <xsl:element name="end-do-stmt" namespace="http://g95-xml.sourceforge.net/">
         <xsl:element name="L" namespace="http://g95-xml.sourceforge.net/">
           <xsl:text>&#10;</xsl:text>
         </xsl:element>
         <xsl:element name="c" namespace="http://g95-xml.sourceforge.net/">
           <xsl:text>enddo</xsl:text>
         </xsl:element>
       </xsl:element>
     </xsl:element>
   </xsl:element>
</xsl:template>

</xsl:stylesheet>
