import sys
from os.path import expanduser
import graphviz
import copy
import collections

class Topology:

    def __init__(self, ranksInfo):
        """
        :param nRanks: number of ranks 
        :param nChannels: number of rings in the topology
        """
        self.colors = ["red", "blue", "green", 'cyan']
        self.ringEdges = []
        self.treeEdges = []
        self.ranksInfo = ranksInfo
    
    def setNRanks(self, nRanks):
        self.nRanks = nRanks
    
    def addRingEdge(self, src, dst, eType, cNum):
        """
        :param cNum: channel number, will be colored differently
        :param eType: mark the edge type, P2P/IPC, P2P/direct pointer etc
        """
        self.ringEdges.append([src, dst, eType, cNum])
    
    def addTreeEdge(self, src, dst, cNum):
        """ treeEdge does not have type information currently
        It suppose to use same P2P link type as ring edge of two nodes
        :param cNum: channel number
        """
        self.treeEdges.append([src, dst, cNum])
    
    def _cleanEdges(self):
        """ 1. remove duplicate edge, 
        2. merge bidirectional edge into one, if only src and dst are swapped
        """
        cleanedRingEdges = set()
        for edge in self.ringEdges:
            if tuple(edge) in cleanedRingEdges:
                continue
            cleanedRingEdges.add(tuple(edge))
        self.ringEdges = cleanedRingEdges

        cleanedTreeEdges = set()
        for edge in self.treeEdges:
            if tuple(edge) in cleanedTreeEdges:
                continue
            cleanedTreeEdges.add(tuple(edge))
        self.treeEdges = cleanedTreeEdges

        # merge bi-directional edge
        mergedRingEdges = set()
        for edge in self.ringEdges:
            tmp = list(edge)
            tmp[0], tmp[1] = edge[1], edge[0]
            if tuple(tmp) in self.ringEdges:
                biDirEdge = tmp + ['bi_dir',]
                biDirEdge[0], biDirEdge[1] = min(edge[0], edge[1]), max(edge[0], edge[1])
                biDirEdge = tuple(biDirEdge)
                if biDirEdge not in mergedRingEdges:
                    mergedRingEdges.add(biDirEdge)
            else:
                # no need to merge
                mergedRingEdges.add(edge)
        self.ringEdges = mergedRingEdges

    def graph(self):
        """
        generate graph via graphviz
        """
        g = graphviz.Graph('Network-Topology', filename='topo.gv',
                            node_attr={'nodesep': '2'}, edge_attr={'esep': '2'})
        g.attr(compound='true')

        nameTemplate = "{}"
        for r in self.ranksInfo['ranks']:
            nvmlDev = self.ranksInfo['ranks'][r]
            nodeName = nameTemplate.format(r)
            nodeLabel = "{}[{}]".format(r, nvmlDev)
            g.node(name=nodeName, label=nodeLabel)
        
        # remove duplicates and merge
        self._cleanEdges()

        # ring edges
        groupRingEdges = collections.defaultdict(set)
        otherRingEdges = set()
        for edge in self.ringEdges:
            if self.ranksInfo['hosts'][edge[0]] == self.ranksInfo['hosts'][edge[1]]:
                groupRingEdges[self.ranksInfo['hosts'][edge[0]]].add(edge)
            else:
                otherRingEdges.add(edge)
        for host in groupRingEdges:
            with g.subgraph(name="cluster-{}".format(host), node_attr={'shape':'box'}) as subg:
                for edge in groupRingEdges[host]:
                    srcNode = nameTemplate.format(edge[0])
                    dstNode = nameTemplate.format(edge[1])
                    eLabel = edge[2]
                    cNum = edge[3]
                    if edge[-1] == "bi_dir":
                        subg.edge(srcNode, dstNode, label=eLabel, color=self.colors[cNum], dir='both')
                    else:
                        subg.edge(srcNode, dstNode, label=eLabel, color=self.colors[cNum], dir='forward')
                subg.attr(label=host)

        for edge in otherRingEdges:
            srcNode = nameTemplate.format(edge[0])
            dstNode = nameTemplate.format(edge[1])
            eLabel = edge[2]
            cNum = edge[3]
            
            if edge[-1] == "bi_dir":
                g.edge(srcNode, dstNode, label=eLabel, color=self.colors[cNum], dir='both')
            else:
                g.edge(srcNode, dstNode, label=eLabel, color=self.colors[cNum], dir='forward')
        
        # tree edges
        for edge in self.treeEdges:
            srcNode = nameTemplate.format(edge[0])
            dstNode = nameTemplate.format(edge[1])
            cNum = edge[2]
            g.edge(srcNode, dstNode, color=self.colors[cNum], style='dashed', dir='forward')
        
        g.view()

def getRanksInfo(logfile):
    infos = {
        'nRanks': -1,
        'ranks': {},
        'hosts': {}
    }
    with open(logfile) as ifile:
        for line in ifile:
            line = line.strip('\n')
            if "NCCL INFO comm" in line and line.endswith("Init COMPLETE"):
                fields = line.split(' ')
                nranks = int(fields[fields.index('nranks') + 1])
                rank = int(fields[fields.index('rank') + 1])
                nvmlDev = int(fields[fields.index('nvmlDev') + 1])
                if infos['nRanks'] < 0:
                    infos['nRanks'] = nranks
                infos['ranks'][rank] = nvmlDev

                # get host information, the hostname
                hostname = fields[0].split(':')[1]
                infos['hosts'][rank] = hostname

            if len(infos['ranks']) == infos['nRanks']:
                return infos
    return infos


def isTreeLog(line):
    ret = True if "NCCL INFO Trees" in line else False
    return ret

def isRingLog(line):
    ret = True if "NCCL INFO Ring" in line else False
    return ret

def parseRingLog(line):
    """"""
    # import pdb; pdb.set_trace()
    line = line.strip('\n')
    fields = line.split(' ')
    cNum = int(fields[fields.index('Ring') + 1])
    eType = fields[fields.index('via') + 1]

    src = fields[fields.index('->') - 1]
    dst = fields[fields.index('->') + 1]

    if '[receive]' in fields or '[send]' in fields:
        return int(src), int(dst), eType, cNum
    else:
        srcRank = int(src[:src.index('[')])
        dstRank = int(dst[:dst.index('[')])
        return srcRank, dstRank, eType, cNum

def parseTreeLog(line):
    """"""
    line = line.strip('\n')
    fields = line.split(' ')
    startIdx = fields.index('Trees')
    i = 1
    edges = []
    while startIdx + i < len(fields):
        cNum = fields[startIdx + i]
        cNum = int(cNum[1:len(cNum)-1])
        treeEdges = fields[startIdx + i + 1]
        up, cRank, downs = treeEdges.split('->')
        downs = downs.split('/')
        if up != '-1':
            edges.append([int(up), int(cRank), cNum])
        for d in downs:
            if d != '-1':
                edges.append([int(cRank), int(d), cNum])
        
        i += 2
    return edges


def main():
    """"""
    if len(sys.argv) < 2:
        print("put log file to parse")
        return
    logfile = expanduser(sys.argv[1])

    ranksInfo = getRanksInfo(logfile)
    print(ranksInfo)
    topo = Topology(ranksInfo)

    with open(logfile) as ifile:
        for line in ifile:
            if isRingLog(line):
                src, dst, eType, cNum = parseRingLog(line)
                topo.addRingEdge(src, dst, eType, cNum)
            elif isTreeLog(line):
                treeEdges = parseTreeLog(line)
                for edge in treeEdges:
                    topo.addTreeEdge(*edge)
            else:
                pass
        topo.graph()

if __name__ == "__main__":
    main()