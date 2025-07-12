#!/bin/bash
######################################################################################
# Chad Netwig
# May 26, 2024
#
# DESCRIPTION:
# Default DROP policies: Ensures that any traffic not explicitly allowed is dropped.
# Input chain: Allows HTTP traffic from the client to the server.
# Output chain: Allows HTTP responses from the server back to the client.
# Forward chain: Allows ICMP traffic (ping) between the client and 8.8.8.8.
# NAT rule: Masquerades outgoing traffic from the client to external networks,
#           changing the source IP to the gateway's external IP.
#
######################################################################################

# Enable IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1

# Flush existing iptables rules
sudo iptables -F                     # Flush iptables rules 
sudo iptables -X                     # Delete user-defined chains  

# Setup for whitelist
sudo iptables -P INPUT DROP          # Set default policy for INPUT chain 
sudo iptables -P OUTPUT DROP         # Set default policy for OUTPUT chain
sudo iptables -P FORWARD DROP        # Set default policy for FORWARD chain

# Forward Chain (icmp)
sudo iptables -A FORWARD -p icmp -s 10.0.2.7 -d 8.8.8.8 -o enp0s3 -j ACCEPT   # allow ping request to 8.8.8.8 from client
sudo iptables -A FORWARD -p icmp -s 8.8.8.8 -d 10.0.2.7 -i enp0s3 -j ACCEPT   # allow ping response to client from 8.8.8.8


# Input Chain
sudo iptables -A INPUT -p tcp --dport 80 -s 10.0.2.7 -d 10.0.2.4 -j ACCEPT  # allow incoming TCP traffic on port 80 from 10.0.2.7 to 10.0.2.4

# Output Chain
sudo iptables -A OUTPUT -p tcp --sport 80 -s 10.0.2.4 -d 10.0.2.7 -j ACCEPT  # allow outgoing TCP traffic on port 80 from 10.0.2.4 to 10.0.2.7

# NAT - Postrouting
sudo iptables -t nat -A POSTROUTING -o enp0s8 -d 8.8.8.8 -j MASQUERADE  # apply NAT to outgoing traffic to 8.8.8.8 on interface enp0s8


# Log dropped packets for debugging
sudo iptables -A FORWARD -j LOG --log-prefix "IPTables-Dropped: " --log-level 4

# Display iptables rules
sudo iptables -L -v  # List all iptables rules
sudo iptables -t nat -L -v  # List NAT iptables rules

