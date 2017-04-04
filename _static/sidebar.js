/*
 * sidebar.js
 * ~~~~~~~~~~
 *
 * This script makes the Sphinx sidebar collapsible.
 *
 * .sphinxsidebar contains .sphinxsidebarwrapper.  This script adds
 * in .sphixsidebar, after .sphinxsidebarwrapper, the #sidebarbutton
 * used to collapse and expand the sidebar.
 *
 * When the sidebar is collapsed the .sphinxsidebarwrapper is hidden
 * and the width of the sidebar and the margin-left of the document
 * are decreased. When the sidebar is expanded the opposite happens.
 * This script saves a per-browser/per-session cookie used to
 * remember the position of the sidebar among the pages.
 * Once the browser is closed the cookie is deleted and the position
 * reset to the default (expanded).
 *
 * :copyright: Copyright 2007-2011 by the Sphinx team, see AUTHORS.
 * :license: BSD, see LICENSE for details.
 *
 */

$(function() {
  // global elements used by the functions.
  // the 'sidebarbutton' element is defined as global after its
  // creation, in the add_sidebar_button function
  var bodywrapper = $('.bodywrapper');
  var content = $('.content');
  var sidebar = $('.sphinxsidebar');
  var sidebarwrapper = $('.sphinxsidebarwrapper');

  // for some reason, the document has no sidebar; do not run into errors
  if (!sidebar.length) return;

  // colors used by the current theme
  var dark_color = $('.related').css('background-color');
  var light_color = $('.footer').css('color');

  function sidebar_is_collapsed() {
    return sidebarwrapper.is(':not(:visible)');
  }

  function toggle_sidebar() {
    if (sidebar_is_collapsed())
      expand_sidebar();
    else
      collapse_sidebar();
  }

  function collapse_sidebar() {
    sidebarwrapper.hide();
    sidebar.width('1.5%');
    content.css('margin-left', '3%');
    sidebarbutton.css({
        'margin-left': '0',
        'height': bodywrapper.height(),
	'left': '0'
    });
    sidebarbutton.find('span').text('»');
    sidebarbutton.attr('title', _('Expand sidebar'));
    document.cookie = 'sidebar=collapsed';
  }

  function expand_sidebar() {
    content.css('margin-left', '16%');
    sidebar.width('15%');
    sidebarwrapper.show();
    sidebarbutton.css({
        'margin-left': '90%',
        'height': bodywrapper.height(),
	'left': '0'
    });
    sidebarbutton.find('span').text('«');
    sidebarbutton.attr('title', _('Collapse sidebar'));
    document.cookie = 'sidebar=expanded';
  }

  function add_sidebar_button() {
    sidebarwrapper.css({
        'float': 'left' ,
        'margin-right': '0',
        'width': '90%'
    });
    // create the button
    sidebar.append(
        '<div id="sidebarbutton"><span>&laquo;</span></div>'
    );
    var sidebarbutton = $('#sidebarbutton');
    sidebarbutton.find('span').css({
        'display': 'block',
        'margin-top': '20%',
	    'margin-left': '0em'
    });

    sidebarbutton.click(toggle_sidebar);
    sidebarbutton.attr('title', _('Collapse sidebar'));
    sidebarbutton.css({
        'border-left': '1px solid ' + dark_color,
	'border-top-left-radius' : '1em',
	'border-bottom-left-radius' : '1em',
        'font-size': '1.2em',
        'cursor': 'pointer',
        'height': bodywrapper.height(),
        'padding-top': '1em',
        'margin-left': '90%'
    });

    sidebarbutton.hover(
      function () {
          $(this).css('background-color', '#D0D0D0');
      },
      function () {
          $(this).css('background-color', '#F6F6F6');
      }
    );
  }

  function set_position_from_cookie() {
    if (!document.cookie)
      return;
    var items = document.cookie.split(';');
    for(var k=0; k<items.length; k++) {
      var key_val = items[k].split('=');
      var key = key_val[0];
      if (key == 'sidebar') {
        var value = key_val[1];
        if ((value == 'collapsed') && (!sidebar_is_collapsed()))
          collapse_sidebar();
        else if ((value == 'expanded') && (sidebar_is_collapsed()))
          expand_sidebar();
      }
    }
  }

  add_sidebar_button();
  var sidebarbutton = $('#sidebarbutton');
  // set_position_from_cookie();
  // Uncomment the above to have cookies remember sidebar position
  // Causes problems on pages without sidebar like index
});
