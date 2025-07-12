######################################################################################
# Chad Netwig
# June 23, 2024
#
# DESCRIPTION:
# 1. Created a Mininet-based topology with 4 container hosts and one controller switch
#    - Add a link from controller 1 to switch 1. 
#    - Add a link from switch 1 to container 1. 
#    - Add a link from switch 1 to container 2. 
#    - Add a link from switch 1 to container 3. 
#    - Add a link from switch 1 to container 4. 
# 2. Made the interfaces up and assigned IP addresses to interfaces
#    of container hosts. 
#    - Assign IP address 192.168.2.10 to container host #1. 
#    - Assign IP address 192.168.2.20 to container host #2. 
#    - Assign IP address 192.168.2.30 to container host #3. 
#    - Assign IP address 192.168.2.40 to container host #4. 
#
######################################################################################

from mininet.net import Containernet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.link import TCLink

# Create Mininet object
#net = Mininet(controller=RemoteController, switch=OVSSwitch)

# Create Containernet object
net = Containernet(controller=RemoteController, switch=OVSSwitch)

# Add remote controllers (ensure POX controllers running on localhost)
c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)
#c1 = net.addController('c1', controller=RemoteController, ip='127.0.0.1', port=6655)

# Add Docker containers with IP assignments and MAC addresses
h1 = net.addDocker('h1', ip='192.168.2.10', dimage="ubuntu:trusty", mac="00:00:00:00:00:01")
h2 = net.addDocker('h2', ip='192.168.2.20', dimage="ubuntu:trusty", mac="00:00:00:00:00:02")
h3 = net.addDocker('h3', ip='192.168.2.30', dimage="ubuntu:trusty", mac="00:00:00:00:00:03")
h4 = net.addDocker('h4', ip='192.168.2.40', dimage="ubuntu:trusty", mac="00:00:00:00:00:04")

# Add a switch with MAC learning enabled
s1 = net.addSwitch('s1', switch=OVSSwitch, mac=True)

# Add links from switch to containers
net.addLink(h1, s1)
net.addLink(h2, s1)
net.addLink(h3, s1)
net.addLink(h4, s1)

# Build network
net.build()

# Start switch s1 with binding controller c0 to it
s1.start([c0])

# Start CLI
CLI(net)

