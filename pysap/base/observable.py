# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


class SignalObject(object):
    """ Dummy class for signals.
    """
    pass


class Observable(object):
    """ Base class for observable classes.

    This class defines a simple interface to add or remove observers
    on an object.
    """

    def __init__(self, signals):
        """ Initilize the Observable class.

        Parameters
        ----------
        signals: list of str
            the allowed signals.
        """
        # Define class parameters
        self._allowed_signals = []
        self._observers = {}

        # Set allowed signals
        for signal in signals:
            self._allowed_signals.append(signal)
            self._observers[signal] = []

        # Set a lock option to avoid multiple observer notifications
        self._locked = False

    def add_observer(self, signal, observer):
        """ Add an observer to the object.
        Raise an exception if the signal is not allowed.

        Parameters
        ----------
        signal: str
            a valid signal.
        observer: @func
            a function that will be called when the signal is emitted.
        """
        self._is_allowed_signal(signal)
        self._add_observer(signal, observer)

    def remove_observer(self, signal, observer):
        """ Remove an observer from the object.
        Raise an eception if the signal is not allowed.

        Parameters
        ----------
        signal: str
            a valid signal.
        observer: @func
            an obervation function to be removed.
        """
        self._is_allowed_event(signal)
        self._remove_observer(signal, observer)

    def notify_observers(self, signal, **kwargs):
        """ Notify observers of a given signal.

        Parameters
        ----------
        signal: str
            a valid signal.
        kwargs: dict
            the parameters that will be sent to the observers.

        Returns
        -------
        out: bool
            Fasle if a notification is in progress, otherwise True.
        """
        # Chack if a notification if in progress
        if self._locked:
            return False

        # Set the lock
        self._locked = True

        # Create a signal object
        signal_to_be_notified = SignalObject()
        setattr(signal_to_be_notified, "object", self)
        setattr(signal_to_be_notified, "signal", signal)
        for name, value in kwargs.items():
            setattr(signal_to_be_notified, name, value)

        # Notify all the observers
        for observer in self._observers[signal]:
            observer(signal_to_be_notified)

        # Unlock the notification process
        self._locked = False

    ######################################################################
    # Properties
    ######################################################################

    def _get_allowed_signals(self):
        """ Events allowed for the current object.
        """
        return self._allowed_signals

    allowed_signals = property(_get_allowed_signals)

    ######################################################################
    # Private interface
    ######################################################################

    def _is_allowed_signal(self, signal):
        """ Check if a signal is valid.
        Raise an exception if the signal is not allowed.

        Parameters
        ----------
        signal: str
            a signal.
        """
        if signal not in self._allowed_signals:
            raise Exception("Signal '{0}' is not allowed for '{1}'.".format(
                signal, type(self)))

    def _add_observer(self, signal, observer):
        """ Assocaite an observer to a valid signal.

        Parameters
        ----------
        signal: str
            a valid signal.
        observer: @func
            an obervation function.
        """
        if observer not in self._observers[signal]:
            self._observers[signal].append(observer)

    def _remove_observer(self, signal, observer):
        """ Remove an observer to a valid signal.

        Parameters
        ----------
        signal: str
            a valid signal.
        observer: @func
            an obervation function to be removed.
        """
        if observer in self._observers[signal]:
            self._observers[signal].remove(observer)
