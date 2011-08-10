#ifndef _XML_DUMP_H
#define _XML_DUMP_H

#ifndef INTF
#include <ctype.h>
#include "g95.h"
#include "xml-dump-decl.h"
#include "xml-dump-attr.h"
#include "xml-dump-body.h"
#include "xml-string.h"
#endif

#define nop do{}while(0)

#undef G95X_DUMP_LOCUS



#ifdef G95X_DUMP_ATTR

#define G95X_EX(x) do { if (g95x_option.xml_example) g95x_xml_example (x); } while (0)

#define G95X_DUMP_FCT( type, arg ) g95x_dump_##type##_attr( arg )

#define G95X_DUMP_ATTR_FCT( type, arg ) nop

#define G95X_ADD_SCT( name ) nop
#define G95X_END_SCT()       nop

#define G95X_DUMP_SUBOBJ_NA( name, type, arg ) nop

#define G95X_DUMP_SUBOBJ_AN( type, arg ) nop
#define G95X_DUMP_SRCA( loc ) nop
#define G95X_DUMP_SRCB( loc ) nop


#define G95X_ADD_OBJ_NI( name, arg )      nop
#define G95X_ADD_OBJ( name, arg )         nop
#define G95X_ADD_ALS_AN( name, als, arg ) nop
#define G95X_ADD_ALS( name, als, arg )    nop
#define G95X_END_OBJ( type, arg )      g95x_end_att1()

#define G95X_ADD_OBJ_T_NI( name, T, obj )   nop
#define G95X_ADD_OBJ_T( name, T, obj )      nop
#define G95X_ADD_ALS_T( name, T, als, obj ) nop


#define G95X_ADD_LST( name )          nop
#define G95X_END_LST()                nop
#define G95X_PSH_LST_STR( val )       nop
#define G95X_PSH_LST_OBJ( val )       nop
#define G95X_PSH_LST_INT( val )       nop
#define G95X_PSH_LST_CHR( val )       nop

#define G95X_ADD_ATT_STR( key, val )      g95x_add_att1_str( key, val )
#define G95X_ADD_ATT_INT( key, val )      g95x_add_att1_int( key, val )
#define G95X_ADD_ATT_CHR( key, val )      g95x_add_att1_chr( key, val )
#define G95X_ADD_ATT_OBJ( key, val )      g95x_add_att1_obj( key, val )
#define G95X_ADD_ATT_CST( key, val )      g95x_add_att1_cst( key, val )
#define G95X_ADD_ATT_ALS( key, als, val ) g95x_add_att1_als( key, als, val )

#define G95X_ADD_TXT( key, txt )                nop
#define G95X_PSH_LOC_NA( name, type, arg, lst ) nop
#define G95X_PSH_LOC_AN( type, arg, lst )       nop

#define G95X_DUMP_WHERE( xw ) nop


#endif


#ifdef G95X_DUMP_BODY

#define G95X_EX(x) do {} while (0)

#define G95X_DUMP_FCT( type, arg ) g95x_dump_##type##_body( arg )

#define G95X_DUMP_ATTR_FCT( type, arg ) g95x_dump_##type##_attr( arg )

#define G95X_ADD_SCT( name ) g95x_add_sct1( name )
#define G95X_END_SCT()       g95x_end_sct1()

#define G95X_GET_LOCUS( type, arg ) \
  g95x_get_##type##_locus( arg )

#define G95X_DUMP_SUBOBJ_NA( name, type, arg )      \
  do {                                              \
    g95x_locus *_xw = G95X_GET_LOCUS( type, arg );  \
    if( _xw )                                       \
      G95X_DUMP_SRCA( _xw );                        \
    G95X_ADD_SCT( name );                           \
    G95X_DUMP_FCT( type, arg );                     \
    G95X_END_SCT();                                 \
  } while( 0 )

#define G95X_DUMP_SUBOBJ_AN( type, arg )            \
  do {                                              \
    g95x_locus *_xw = G95X_GET_LOCUS( type, arg );  \
    if( _xw )                                       \
      G95X_DUMP_SRCA( _xw );                        \
    g95x_dump_##type##_body( arg );                 \
  } while( 0 )

#define G95X_DUMP_SRCA( loc ) g95x_dump_source_code( loc, 0, 1 )
#define G95X_DUMP_SRCB( loc ) g95x_dump_source_code( loc, 1, 1 )

#define G95X_ADD_OBJ_NI( name, obj )        \
  do {                                      \
    if (g95x_option.xml_watch)              \
      g95x_new_watch_id ();                 \
    g95x_add_obj1( name, NULL, 1, obj );    \
  } while (0)
#define G95X_ADD_OBJ( name, obj )           \
  do {                                      \
    if (g95x_option.xml_watch)              \
      g95x_new_watch_id ();                 \
    g95x_add_obj1( name, NULL, 0, obj );    \
  } while (0)
#define G95X_ADD_ALS( name, als, obj )      \
  do {                                      \
    if (g95x_option.xml_watch)              \
      g95x_new_watch_id ();                 \
    g95x_add_als1( name, NULL, als, obj );  \
  } while (0)

#define G95X_ADD_OBJ_T_NI( name, T, obj )        \
  do {                                           \
    if (g95x_option.xml_watch)                   \
      g95x_new_watch_id ();                      \
    g95x_add_obj1( name, T, 1, obj );            \
  } while (0)
#define G95X_ADD_OBJ_T( name, T, obj )           \
  do {                                           \
    if (g95x_option.xml_watch)                   \
      g95x_new_watch_id ();                      \
    g95x_add_obj1( name, T, 0, obj );            \
  } while (0)
#define G95X_ADD_ALS_T( name, T, als, obj )      \
  do {                                           \
    if (g95x_option.xml_watch)                   \
      g95x_new_watch_id ();                      \
    g95x_add_als1( name, T, als, obj );          \
  } while (0)


#define G95X_END_OBJ( type, arg )                   \
  do {                                              \
    g95x_locus *_xw =  G95X_GET_LOCUS( type, arg ); \
    if( _xw )                                       \
      G95X_DUMP_SRCB( _xw );                        \
    g95x_end_obj1();                                \
  } while( 0 )

#define G95X_ADD_LST( name )          g95x_add_lst1( name )
#define G95X_END_LST()                g95x_end_lst1()
#define G95X_PSH_LST_STR( val )       g95x_psh_lst1_str( val )
#define G95X_PSH_LST_OBJ( val )       g95x_psh_lst1_obj( val )
#define G95X_PSH_LST_INT( val )       g95x_psh_lst1_int( val )
#define G95X_PSH_LST_CHR( val )       g95x_psh_lst1_chr( val )

#define G95X_ADD_ATT_STR( key, val )          nop
#define G95X_ADD_ATT_INT( key, val )          nop
#define G95X_ADD_ATT_CHR( key, val )          nop
#define G95X_ADD_ATT_OBJ( key, val )          nop
#define G95X_ADD_ATT_CST( key, val )          nop
#define G95X_ADD_ATT_ALS( key, als, val )     nop

#define G95X_PSH_LOC_NA( name, type, arg, lst ) \
  do {                                                 \
    g95x_item_locus* il = g95x_get_item_locus();       \
    il->next = lst;                                    \
    lst = il;                                          \
    il->xwl = G95X_GET_LOCUS( type, arg );             \
    il->obj = arg;                                     \
    il->key = name;                                    \
    il->dmp = (g95x_dumper)g95x_dump_##type##_body;    \
  } while( 0 )

#define G95X_PSH_LOC_AN( type, arg, lst ) \
  G95X_PSH_LOC_NA( NULL, type, arg, lst )

#ifdef G95X_DUMP_LOCUS
#define G95X_DUMP_WHERE( xw ) G95X_DUMP_SUBOBJ_NA( "where", locus, xw )
#else
#define G95X_DUMP_WHERE( xw ) nop
#endif

#endif


#define g95x_get_abstract_locus( e )                  NULL
#define g95x_get_association_locus( e )               NULL
#define g95x_get_common_locus( e )                    NULL
#define g95x_get_component_locus( e )                 NULL
#define g95x_get_generic_locus( e )                   NULL
#define g95x_get_intf_info_block_uop_locus( e )       NULL
#define g95x_get_intf_info_block_gen_locus( e )       NULL
#define g95x_get_intf_info_block_iop_locus( e )       NULL
#define g95x_get_intrinsic_assignment_base_locus( e ) NULL
#define g95x_get_intrinsic_assignment_locus( e )      NULL
#define g95x_get_intrinsic_gen_locus( e )             NULL
#define g95x_get_intrinsic_operator_base_locus( e )   NULL
#define g95x_get_intrinsic_operator_locus( e )        NULL
#define g95x_get_intrinsic_sym_locus( e )             NULL
#define g95x_get_intrinsic_type_locus( e )            NULL
#define g95x_get_imported_abstract_locus( e )         NULL
#define g95x_get_imported_component_locus( e )        NULL
#define g95x_get_imported_symbol_locus( e )           NULL
#define g95x_get_imported_user_type_locus( e )        NULL
#define g95x_get_locus_locus( e )                     NULL
#define g95x_get_locus_section_locus( e )             NULL
#define g95x_get_symbol_locus( e )                    NULL
#define g95x_get_global_symbol_locus( e )             NULL
#define g95x_get_global_common_locus( e )             NULL
#define g95x_get_user_operator_locus( e )             NULL
#define g95x_get_user_type_locus( e )                 NULL
#define g95x_get_label_locus( e )                     NULL



#define g95x_get_intrinsic_type_name_locus( e )       &((e)->where)
#define g95x_get_common_block_object_locus( e )       &((e)->where)
#define g95x_get_actual_arg_locus( e )                &((e)->x.where)
#define g95x_get_array_spec_locus( e )                &((e)->x.where)
#define g95x_get_coarray_spec_locus( e )              &((e)->x.where)
#define g95x_get_constant_location_locus( e )         (e)
#define g95x_get_contains_block_locus( e )            &((e)->where_block)
#define g95x_get_coshape_spec_locus( e )              &((e)->where)
#define g95x_get_critical_block_locus( e )            &((e)->where_construct)
#define g95x_get_do_block_locus( e )                  &((e)->where_construct)
#define g95x_get_component_decl_locus( e )            &((e)->where)
#define g95x_get_entity_decl_locus( e )               &((e)->where)
#define g95x_get_enumeration_locus( e )               &((e)->where)
#define g95x_get_expr_chst_locus( e )                 ((e)->where)
#define g95x_get_expr_locus( e )                      &((e)->x.where)
#define g95x_get_forall_block_locus( e )              &((e)->where_construct)
#define g95x_get_formal_arg_locus( e )                &((e)->x.where)
#define g95x_get_if_block_locus( e )                  &((e)->where_construct)
#define g95x_get_interface_block_locus( e )           &((e)->where_construct)
#define g95x_get_io_var_locus( e )                    &((e)->x.where)
#define g95x_get_keyword_locus( e )                   &((e)->x.name_where)
#define g95x_get_label_ref_locus( e )                 ((e)->xw)
#define g95x_get_named_constant_def_locus( e )        &((e)->where)
#define g95x_get_namespace_locus( e )                 &((e)->x.where)
#define g95x_get_ref_locus( e )                       &((e)->x.where)
#define g95x_get_rename_locus( e )                    &((e)->x.where)
#define g95x_get_section_subscript_locus( e )         &((e)->ref->u.ar.x.section_where[(e)->index])
#define g95x_get_select_block_locus( e )              &((e)->where_construct)
#define g95x_get_shape_spec_locus( e )                &((e)->as->x.shape_spec_where[(e)->i])
#define g95x_get_statement_locus( e )                 &((e)->where)
#define g95x_get_symbol_ref_locus( e )                ((e)->xw)
#define g95x_get_typespec_locus( e )                  &((e)->x.where)
#define g95x_get_use_name_locus( e )                  &((e)->x.use_where)
#define g95x_get_where_block_locus( e )               &((e)->where_construct)
#define g95x_get_section_cosubscript_locus( e )       &((*e)->x.where)
#define g95x_get_executable_program_locus( e )        &((e)->where)
#define g95x_get_alt_return_locus( e )                ((e)->where)
#define g95x_get_io_control_spec_locus( e )           &((e)->where)
#define g95x_get_attr_spec_locus( e )                 &((e)->where)
#define g95x_get_prefix_spec_locus( e )               &((e)->where)
#define g95x_get_implicit_spec_locus( e )             &((e)->where)
#define g95x_get_letter_spec_locus( e )               &((e)->where)
#define g95x_get_derived_block_locus( e )             &((e)->where_construct)
#define g95x_get_statement_block_locus( e )           &((e)->where_block)
#define g95x_get_user_operator1_ref_locus( e )        &((e)->xwo)
#define g95x_get_intrinsic_operator1_ref_locus( e )   &((e)->xwo)
#define g95x_get_assignment1_ref_locus( e )           &((e)->xwo)
#define g95x_get_component_spec_locus( e )            &((e)->x.where)
#define g95x_get_alloc_locus( e )                     &((e)->x.where)
#define g95x_get_alloc_spec_locus( e )                &((e)->where)
#define g95x_get_co_alloc_spec_locus( e )             &((e)->where)
#define g95x_get_stop_code_locus( e )                 &((e)->where)
#define g95x_get_position_spec_unit_locus( e )        &((e)->x.where)



#endif
