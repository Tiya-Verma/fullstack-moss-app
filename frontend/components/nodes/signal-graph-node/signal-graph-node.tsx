import { Card } from '@/components/ui/card';
import { getIncomers, Handle, Node, Position, useReactFlow, useNodeId } from '@xyflow/react';
import SignalGraphPreview from './signal-graph-preview';
import useWebsocket from '@/hooks/useWebsocket';

import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from '@/components/ui/dialog';
import SignalGraphView from './signal-graph-full';

export default function SignalGraphNode() {
    const nodeId = useNodeId();
    const { getNode } = useReactFlow();
    const node = nodeId ? getNode(nodeId) : null;
    const droppedOnCanvas = node?.data?.droppedOnCanvas ?? false;

    const { renderData } = useWebsocket(20, 10);

    const processedData = renderData.map((item) => ({
        time: String(item.time ?? ''),
        signal1: Number(item.signals?.[0] ?? 0),
        signal2: Number(item.signals?.[1] ?? 0),
        signal3: Number(item.signals?.[2] ?? 0),
        signal4: Number(item.signals?.[3] ?? 0),
        signal5: Number(item.signals?.[4] ?? 0),
    }));

    const { getEdges, getNodes } = useReactFlow();
    const nodes = getNodes();
    const edges = getEdges();

    const areNodeTypesConnected = (nodes: any[], edges: any[], sourceTypes: string[], targetType: string) => {
        const nodeMap = Object.fromEntries(nodes.map((n) => [n.id, n]));
      
        return edges.some((edge) => {
          const sourceNode = nodeMap[edge.source];
          const targetNode = nodeMap[edge.target];
      
          return (sourceTypes.includes(sourceNode?.type) && targetNode?.type === targetType);
        });
    };

    const connected = areNodeTypesConnected(nodes, edges, ['source-node', 'filter-node'], 'signal-graph-node');

    if (connected) {
        console.log('At least one inputNode is connected to an outputNode');
    }       

    const previewData = connected ? processedData : [];

    return (
        <Card>
            <Handle type="target" position={Position.Left} />
            <Handle type="source" position={Position.Right} />
            <Dialog>
                <DialogTrigger>
                    <div className="w-[400px] h-[400px]">
                        <SignalGraphPreview data={previewData} />
                    </div>
                </DialogTrigger>
                <DialogContent className="w-[80vw] h-[80vh] max-w-none max-h-none">
                    <DialogHeader>
                        <DialogTitle>Signal Graph</DialogTitle>
                        <DialogDescription>
                            Here is a preview of the signal graph.
                        </DialogDescription>
                    </DialogHeader>
                    <Card>
                        <div className="w-full h-full">
                            <SignalGraphView data={previewData} droppedOnCanvas={!!droppedOnCanvas} />
                        </div>
                    </Card>
                </DialogContent>
            </Dialog>
        </Card>
    );
}