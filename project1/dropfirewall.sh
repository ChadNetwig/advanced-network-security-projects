#!/bin/bash
# Flush all existing rules
sudo iptables -F

# Delete all user-defined chains
sudo iptables -X

# Set default policies to DROP
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT DROP
