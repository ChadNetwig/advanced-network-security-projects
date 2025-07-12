######################################################################################
# Chad Netwig
# June 23, 2024
#
# Changes made throughout the code with comment prepended with my initials "CLN":
# CLN - Added defaultdict import for port security table initialization.
# CLN - Modified __init__ method to initialize port_table dictionary struct
# CLN - Added block_mac method to block MAC addresses when IP spoofing is detected.
# CLN - Modified _handle_PacketIn method to check for IP spoofing and call block_mac.
# CLN - Added console logging for blocked MAC addresses due to IP spoofing.
# CLN - BONUS Points: # Added functionality to block IP addresses that change 
#       their associated MAC addresses due to MAC spoofing.
#
# DESCRIPTION:
# 1. Implemented port_table as a defaultdict for efficient MAC-to-IP mapping storage.
# 2. Added IP spoofing detection to block MAC addresses if IP changes detected.
# 3. Console logging provides visibility into MAC address blocks due to IP spoofing.
######################################################################################

from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.revent import *
from pox.lib.util import dpidToStr
from pox.lib.addresses import EthAddr
import os
''' New imports here ... '''
from collections import defaultdict  # CLN: New import for port security
import argparse
import csv
from pox.lib.packet.ethernet import ethernet, ETHER_BROADCAST
from pox.lib.addresses import IPAddr
import pox.lib.packet as pkt
from pox.lib.packet.arp import arp
from pox.lib.packet.ipv4 import ipv4
from pox.lib.packet.icmp import icmp

log = core.getLogger()
priority = 50000

l2config = "l2firewall.config"
l3config = "l3firewall.config"


class Firewall (EventMixin):

    def __init__ (self, l2config, l3config):
        self.listenTo(core.openflow)
        self.disabled_MAC_pair = []  # Shore a tuple of MAC pair which will be installed
        self.port_table = defaultdict(lambda: None)  # CLN: Initialize port secuirty table with defaultdict
	self.ip_table = defaultdict(lambda: None)  # CLN: Initialize IP table to track IP to MAC mappings
	self.fwconfig = list()

        '''
        Read the CSV file
        '''
        if l2config == "":
            l2config = "l2firewall.config"
            
        if l3config == "":
            l3config = "l3firewall.config"
        
        with open(l2config, 'rb') as rules:
            csvreader = csv.DictReader(rules)
            for line in csvreader:
                if line['mac_0'] != 'any':
                    mac_0 = EthAddr(line['mac_0'])
                else:
                    mac_0 = None

                if line['mac_1'] != 'any':
                    mac_1 = EthAddr(line['mac_1'])
                else:
                    mac_1 = None

                self.disabled_MAC_pair.append((mac_0, mac_1))

	with open(l3config) as csvfile:
		log.debug("Reading log file !")
		self.rules = csv.DictReader(csvfile)
		for row in self.rules:
			log.debug("Saving individual rule parameters in rule dict !")
			s_ip = row['src_ip']
			d_ip = row['dst_ip']
			s_port = row['src_port']
			d_port = row['dst_port']
			print "src_ip, dst_ip, src_port, dst_port", s_ip,d_ip,s_port,d_port

        log.debug("Enabling Firewall Module")


    def replyToARP(self, packet, match, event):
        r = arp()
        r.opcode = arp.REPLY
        r.hwdst = match.dl_src
        r.protosrc = match.nw_dst
        r.protodst = match.nw_src
        r.hwsrc = match.dl_dst
        e = ethernet(type=packet.ARP_TYPE, src=r.hwsrc, dst=r.hwdst)
        e.set_payload(r)
        msg = of.ofp_packet_out()
        msg.data = e.pack()
        msg.actions.append(of.ofp_action_output(port=of.OFPP_IN_PORT))
        msg.in_port = event.port
        event.connection.send(msg)


    def allowOther(self, event):
        msg = of.ofp_flow_mod()
        match = of.ofp_match()
        action = of.ofp_action_output(port=of.OFPP_NORMAL)
        msg.actions.append(action)
        event.connection.send(msg)


    def installFlow(self, event, offset, srcmac, dstmac, srcip, dstip, sport, dport, nwproto):
        msg = of.ofp_flow_mod()
        match = of.ofp_match()
        if srcip is not None:
            match.nw_src = IPAddr(srcip)
        if dstip is not None:
            match.nw_dst = IPAddr(dstip)
        match.nw_proto = int(nwproto)
        match.dl_src = srcmac
        match.dl_dst = dstmac
        match.tp_src = sport
        match.tp_dst = dport
        match.dl_type = pkt.ethernet.IP_TYPE
        msg.match = match
        msg.hard_timeout = 0
        msg.idle_timeout = 200
        msg.priority = priority + offset
        event.connection.send(msg)


    def replyToIP(self, packet, match, event, fwconfig):
        srcmac = str(match.dl_src)
        dstmac = str(match.dl_src)
        sport = str(match.tp_src)
        dport = str(match.tp_dst)
        nwproto = str(match.nw_proto)

        with open(l3config) as csvfile:
            log.debug("Reading log file !")
            self.rules = csv.DictReader(csvfile)
            for row in self.rules:
                prio = row['priority']
                srcmac = row['src_mac']
                dstmac = row['dst_mac']
                s_ip = row['src_ip']
                d_ip = row['dst_ip']
                s_port = row['src_port']
                d_port = row['dst_port']
                nw_proto = row['nw_proto']
            
                log.debug("You are in original code block ...")
                srcmac1 = EthAddr(srcmac) if srcmac != 'any' else None
                dstmac1 = EthAddr(dstmac) if dstmac != 'any' else None
                s_ip1 = s_ip if s_ip != 'any' else None
                d_ip1 = d_ip if d_ip != 'any' else None
                s_port1 = int(s_port) if s_port != 'any' else None
                d_port1 = int(d_port) if d_port != 'any' else None
                prio1 = int(prio) if prio != None else priority
                if nw_proto == "tcp":
                    nw_proto1 = pkt.ipv4.TCP_PROTOCOL
                elif nw_proto == "icmp":
                    nw_proto1 = pkt.ipv4.ICMP_PROTOCOL
                    s_port1 = None
                    d_port1 = None
                elif nw_proto == "udp":
                    nw_proto1 = pkt.ipv4.UDP_PROTOCOL
                else:
                    log.debug("PROTOCOL field is mandatory, Choose between ICMP, TCP, UDP")
                print (prio1, s_ip1, d_ip1, s_port1, d_port1, nw_proto1)
                self.installFlow(event, prio1, srcmac1, dstmac1, s_ip1, d_ip1, s_port1, d_port1, nw_proto1)
        self.allowOther(event)


    '''
     CLN Description:
        This custom method constructs an OpenFlow message to install a flow entry
        that matches on the specified MAC address and drops all packets
        from that MAC address by sending them to the OFPP_NONE port.

        It logs the action to the POX console when invoked, providing visibility
        into MAC address blocks due to detected IP spoofing attempts.
    
    def block_mac(self, event, mac):
        log.info("Blocking MAC address %s due to detected malicious activity", mac)
        msg = of.ofp_flow_mod()
        match = of.ofp_match()
        match.dl_src = mac
        msg.match = match
        msg.priority = 65535  # Set the highest priority
        msg.hard_timeout = 0
        msg.idle_timeout = 0
        msg.actions.append(of.ofp_action_output(port=of.OFPP_NONE))  # Block the MAC
        event.connection.send(msg)
        log.info("MAC address %s blocked", mac)
    '''

    def block_mac(self, event, mac):
        msg = of.ofp_flow_mod()
        match = of.ofp_match()
        match.dl_src = mac
        msg.match = match
        msg.priority = 65535
        event.connection.send(msg)

        # Create a new flow_mod message to drop packets matching the MAC address
        msg_drop = of.ofp_flow_mod()
        msg_drop.match = match
        msg_drop.priority = 65535
        msg_drop.actions.append(of.ofp_action_output(port=of.OFPP_NONE))
        event.connection.send(msg_drop)

        # Log the action
        log.info("Blocked MAC address %s due to IP spoofing", mac)


    # CLN: block IP addresses that change their associated MAC addresses due to MAC spoofing (Bonus points)
    def block_ip(self, event, ip):
        msg = of.ofp_flow_mod()
        match = of.ofp_match()
        match.nw_src = IPAddr(ip)
        match.dl_type = pkt.ethernet.IP_TYPE  # Ensure the match is for IP packets
        msg.match = match
        msg.priority = 65535
        event.connection.send(msg)

        # CLN: Create a new flow_mod message to drop packets matching the IP address
        msg_drop = of.ofp_flow_mod()
        msg_drop.match = match
        msg_drop.priority = 65535  # CLN: Highest priority
        msg_drop.actions.append(of.ofp_action_output(port=of.OFPP_NONE))
        event.connection.send(msg_drop)

        # CLN: Log the action
        log.info("Blocked IP address %s due to MAC spoofing", ip)


    def _handle_ConnectionUp(self, event):
        ''' Add your logic here ... '''

        '''
        Iterate through the disabled_MAC_pair array, and for each
        pair, install a rule in each OpenFlow switch to disable traffic
        '''
        self.connection = event.connection

        for (source, destination) in self.disabled_MAC_pair:
            message = of.ofp_flow_mod() # OpenFlow massage. Instructs a switch to install a flow
            match = of.ofp_match()
            match.dl_src = source
            match.dl_dst = destination
            message.priority = 65535
            message.match = match
            event.connection.send(message)

        log.debug("Firewall rules installed on %s", dpidToStr(event.dpid))


    # CLN: Updated method to handle MAC blocking due to IP spoofing and IP blocking due to MAC spoofing
    def _handle_PacketIn(self, event):
        packet = event.parsed
        match = of.ofp_match.from_packet(packet)

        if match.dl_type == packet.ARP_TYPE and match.nw_proto == arp.REQUEST:
            self.replyToARP(packet, match, event)

        if match.dl_type == packet.IP_TYPE:
            ip_packet = packet.payload
            if ip_packet.protocol == ip_packet.TCP_PROTOCOL:
                log.debug("TCP it is !")

	    # Extract source MAC address and source IP address from the incoming packet
            src_mac = str(match.dl_src) # CLN: Extract source MAC address
            src_ip = str(ip_packet.srcip) # CLN: Extract source IP address
	
	    self.replyToIP(packet, match, event, self.rules)

	    # CLN: Added functionality for MAC blocking due to IP spoofing
            if self.port_table[src_mac] is None:
                # If MAC address not in port_table, add it with current src_ip
                self.port_table[src_mac] = src_ip # CLN: Initialize port_table entry with current src_ip
            else:
                # If MAC address already in port_table, check if src_ip matches
                if self.port_table[src_mac] != src_ip:
		    # CLN: Log IP spoofing detection
                    log.info("Blocked MAC address %s due to IP spoofing. IP changed from %s to %s", src_mac, self.port_table[src_mac], src_ip)  
                    self.block_mac(event, EthAddr(src_mac))  # CLN: Block MAC if IP spoofed

            # CLN: Added functionality for IP blocking due to MAC spoofing (BONUS)
            if self.ip_table[src_ip] is None:
                # If IP address not in ip_table, add it with current src_mac
                self.ip_table[src_ip] = src_mac  # CLN: Initialize ip_table entry with current src_mac
            else:
                # If IP address already in ip_table, check if src_mac matches
                if self.ip_table[src_ip] != src_mac:
    		    # CLN: Log MAC spoofing detection
                    log.info("Blocked IP address %s due to MAC spoofing. MAC changed from %s to %s", src_ip, self.ip_table[src_ip], src_mac)  
                    self.block_ip(event, IPAddr(src_ip))  # CLN: Block IP if MAC spoofed

            self.allowOther(event) # CLN: Allow other traffic to pass through after processing



def launch(l2config="l2firewall.config", l3config="l3firewall.config"):
    '''
    Starting the Firewall module
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--l2config', action='store', dest='l2config',
                        help='Layer 2 config file', default='l2firewall.config')
    parser.add_argument('--l3config', action='store', dest='l3config',
                        help='Layer 3 config file', default='l3firewall.config')
    core.registerNew(Firewall, l2config, l3config)


