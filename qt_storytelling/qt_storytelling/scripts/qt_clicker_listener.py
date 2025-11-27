#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qt_clicker_listener.py  (keyboard PageUp/PageDown version)

Instead of reading /dev/input, this node reads from the *terminal* stdin.
Your presenter clicker behaves like a keyboard and sends:

  PageUp   -> escape sequence: "\x1b[5~"   (shown as ^[[5~)
  PageDown -> escape sequence: "\x1b[6~"   (shown as ^[[6~)

We map:
  PageUp   -> "prev"
  PageDown -> "next"

and publish them on:
  /qt_clicker/button   (std_msgs/String)

IMPORTANT:
  - Run this in a normal terminal (not via sudo).
  - That terminal must stay focused for the clicker to work.
  - Do NOT put this node inside a roslaunch file, because it needs stdin.
"""

import sys
import select
import termios
import tty

import rospy
from std_msgs.msg import String


def main():
    rospy.init_node("qt_clicker_listener", anonymous=False)
    pub = rospy.Publisher("/qt_clicker/button", String, queue_size=10)

    fd = sys.stdin.fileno()
    # Save old terminal settings and switch to cbreak (raw-ish) mode
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    rospy.loginfo("")
    rospy.loginfo("qt_clicker_listener (keyboard mode) started.")
    rospy.loginfo("This terminal MUST stay active and focused.")
    rospy.loginfo("Press your presenter buttons here:")
    rospy.loginfo("  PageUp   -> publishes 'prev' on /qt_clicker/button")
    rospy.loginfo("  PageDown -> publishes 'next' on /qt_clicker/button")
    rospy.loginfo("Ctrl+C to stop.")
    rospy.loginfo("")

    buf = ""
    rate = rospy.Rate(100)

    try:
        while not rospy.is_shutdown():
            # Non-blocking read from stdin
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if rlist:
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                buf += ch

                # Detect full escape sequences:
                #   PageUp   = ESC [ 5 ~   -> "\x1b[5~"
                #   PageDown = ESC [ 6 ~   -> "\x1b[6~"
                if buf.endswith("\x1b[5~"):
                    msg = String(data="prev")
                    pub.publish(msg)
                    rospy.loginfo("CLICKER: PageUp -> 'prev'")
                    buf = ""
                elif buf.endswith("\x1b[6~"):
                    msg = String(data="next")
                    pub.publish(msg)
                    rospy.loginfo("CLICKER: PageDown -> 'next'")
                    buf = ""
                else:
                    # Keep buffer small (we only care about the last few chars)
                    if len(buf) > 5:
                        buf = buf[-5:]

            rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal settings no matter what
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        rospy.loginfo("qt_clicker_listener exiting, terminal restored.")


if __name__ == "__main__":
    main()

