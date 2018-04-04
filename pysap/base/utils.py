# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Modules that defines usefull tools.
"""


def with_metaclass(meta, *bases):
    """ Function from jinja2/_compat.py.

    License: BSD.

    Use it like this::

        class BaseForm(object):
            pass

        class FormType(type):
            pass

        class Form(with_metaclass(FormType, BaseForm)):
            pass

    This requires a bit of explanation: the basic idea is to make a
    dummy metaclass for one level of class instantiation that replaces
    itself with the actual metaclass.  Because of internal type checks
    we also need to make sure that we downgrade the custom metaclass
    for one level to something closer to type (that's why __call__ and
    __init__ comes back from type etc.).

    This has the advantage over six.with_metaclass of not introducing
    dummy classes into the final MRO.
    """
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__

        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)
    return metaclass("temporary_class", None, {})


def monkeypatch(klass, methodname=None):
    """ Decorator extending class with the decorated callable.

    >>> class A:
    ...     pass
    >>> @monkeypatch(A)
    ... def meth(self):
    ...     return 12
    ...
    >>> a = A()
    >>> a.meth()
    12
    >>> @monkeypatch(A, 'foo')
    ... def meth(self):
    ...     return 12
    ...
    >>> a.foo()
    12

    Parameters
    ----------
    klass: class object
        the class to be decorated.
    methodname: str, default None
        the name of the decorated method. If None, use the function name.

    Returns
    -------
    decorator: callable
        the decorator.
    """
    def decorator(func):
        try:
            name = methodname or func.__name__
        except AttributeError:
            raise AttributeError(
                "{0} has no __name__ attribute: you should provide an "
                "explicit 'methodname'".format(func))
        setattr(klass, name, func)
        return func
    return decorator
