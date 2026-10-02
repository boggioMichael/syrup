"""A stand-in for xdg-desktop-portal's ScreenCast interface, for testing
Wayland capture without a desktop: it "shares" the PipeWire node named by
argv[1] as the window the user picked, and appends each SelectSources
call's options to the JSON-lines log at argv[2]."""

import json
import socket
import sys

import dbus
import dbus.service
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib

BUS = "org.freedesktop.portal.Desktop"
PATH = "/org/freedesktop/portal/desktop"
SCREEN_CAST = "org.freedesktop.portal.ScreenCast"
REQUEST = "org.freedesktop.portal.Request"
SESSION = "org.freedesktop.portal.Session"


class Request(dbus.service.Object):
    @dbus.service.signal(REQUEST, signature="ua{sv}")
    def Response(self, code, results):
        pass


class Session(dbus.service.Object):
    @dbus.service.method(SESSION)
    def Close(self):
        self.remove_from_connection()


class ScreenCast(dbus.service.Object):
    def __init__(self, bus, node, log):
        super().__init__(bus, PATH)
        self.bus, self.node, self.log, self.started = bus, node, log, 0

    def respond(self, sender, options, results):
        path = f"{PATH}/request/{sender[1:].replace('.', '_')}/{options['handle_token']}"
        request = Request(self.bus, path)
        # The reply to the call goes first, then the Response, as the portal does.
        GLib.idle_add(lambda: (request.Response(0, results), request.remove_from_connection()) and False)
        return dbus.ObjectPath(path)

    @dbus.service.method(SCREEN_CAST, in_signature="a{sv}", out_signature="o", sender_keyword="sender")
    def CreateSession(self, options, sender):
        handle = f"{PATH}/session/{sender[1:].replace('.', '_')}/{options['session_handle_token']}"
        Session(self.bus, handle)
        return self.respond(sender, options, {"session_handle": handle})

    @dbus.service.method(SCREEN_CAST, in_signature="oa{sv}", out_signature="o", sender_keyword="sender")
    def SelectSources(self, session, options, sender):
        with open(self.log, "a") as log:
            log.write(json.dumps({k: str(v) for k, v in options.items() if k != "handle_token"}) + "\n")
        return self.respond(sender, options, {})

    @dbus.service.method(SCREEN_CAST, in_signature="osa{sv}", out_signature="o", sender_keyword="sender")
    def Start(self, session, parent, options, sender):
        self.started += 1
        stream = dbus.Struct((dbus.UInt32(self.node), {"source_type": dbus.UInt32(2, variant_level=1)}), signature="ua{sv}")
        return self.respond(
            sender,
            options,
            {"streams": dbus.Array([stream], signature="(ua{sv})"), "restore_token": f"token-{self.started}"},
        )

    @dbus.service.method(SCREEN_CAST, in_signature="oa{sv}", out_signature="h")
    def OpenPipeWireRemote(self, session, options):
        remote = socket.socket(socket.AF_UNIX)
        remote.connect(f"{GLib.get_user_runtime_dir()}/pipewire-0")
        return dbus.types.UnixFd(remote.detach())

    @dbus.service.method(dbus.PROPERTIES_IFACE, in_signature="ss", out_signature="v")
    def Get(self, interface, name):
        return self.GetAll(interface)[name]

    @dbus.service.method(dbus.PROPERTIES_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        return {
            "AvailableSourceTypes": dbus.UInt32(3),
            "AvailableCursorModes": dbus.UInt32(7),
            "version": dbus.UInt32(5),
        }


DBusGMainLoop(set_as_default=True)
bus = dbus.SessionBus()
portal = ScreenCast(bus, int(sys.argv[1]), sys.argv[2])
name = dbus.service.BusName(BUS, bus)
GLib.MainLoop().run()
