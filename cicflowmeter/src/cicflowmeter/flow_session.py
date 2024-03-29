import csv
from collections import defaultdict

import requests
from scapy.sessions import DefaultSession
import threading
from queue import Queue
from .features.context.packet_direction import PacketDirection
from .features.context.packet_flow_key import get_packet_flow_key
from .flow import Flow
import pandas as pd
import tensorflow as tf
from queue import Queue

import numpy as np
EXPIRED_UPDATE = 40
MACHINE_LEARNING_API = "http://localhost:8000/predict"
GARBAGE_COLLECT_PACKETS = 100


class FlowSession(DefaultSession):
    """Creates a list of network flows."""
    
    def __init__(self, *args, **kwargs):
        self.flows = {}
        self.csv_line = 0




        if self.output_mode == "flow":
            output = open(self.output_file, "w")
            self.csv_writer = csv.writer(output)

        self.packets_count = 0

        self.clumped_flows_per_label = defaultdict(list)

        super(FlowSession, self).__init__(*args, **kwargs)

    def toPacketList(self):
        # Sniffer finished all the packets it needed to sniff.
        # It is not a good place for this, we need to somehow define a finish signal for AsyncSniffer
        self.garbage_collect(None)
        return super(FlowSession, self).toPacketList()

    def on_packet_received(self, packet):
        count = 0
        direction = PacketDirection.FORWARD

        if self.output_mode != "flow":
            if "TCP" not in packet:
                return
            elif "UDP" not in packet:
                return

        try:
            # Creates a key variable to check
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))
        except Exception:
            return

        self.packets_count += 1

        # If there is no forward flow with a count of 0
        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))

            if flow is None:
                # If no flow exists create a new flow
                direction = PacketDirection.FORWARD
                flow = Flow(packet, direction)
                packet_flow_key = get_packet_flow_key(packet, direction)
                self.flows[(packet_flow_key, count)] = flow

            elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
                # If the packet exists in the flow but the packet is sent
                # after too much of a delay than it is a part of a new flow.
                expired = EXPIRED_UPDATE
                while (packet.time - flow.latest_timestamp) > expired:
                    count += 1
                    expired += EXPIRED_UPDATE
                    flow = self.flows.get((packet_flow_key, count))

                    if flow is None:
                        flow = Flow(packet, direction)
                        self.flows[(packet_flow_key, count)] = flow
                        break

        elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
            expired = EXPIRED_UPDATE
            while (packet.time - flow.latest_timestamp) > expired:

                count += 1
                expired += EXPIRED_UPDATE
                flow = self.flows.get((packet_flow_key, count))

                if flow is None:
                    flow = Flow(packet, direction)
                    self.flows[(packet_flow_key, count)] = flow
                    break

        flow.add_packet(packet, direction)

        if not self.url_model:
            GARBAGE_COLLECT_PACKETS = 10000

        if self.packets_count % GARBAGE_COLLECT_PACKETS == 0 or (
            flow.duration > 120 and self.output_mode == "flow"
        ):
            self.garbage_collect(packet.time)

    def get_flows(self) -> list:
        return self.flows.values()
    def get_label(self,model,data_queue):
        keys=[' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']
        for i in range(5):
            data=data_queue.get()
            key=data.keys(),",Prediction"
            
            normal=data.values(),",Normal"
            DDoS=data.values(),",DDoS"
            res = {x:data[x] for x in keys}
            res = np.array(list(res.values())).astype(float)
            res = res.reshape(1, 1,res.shape[0])
    
            y_test_pred_prob = model.predict(res, verbose=0)
            print(y_test_pred_prob)
            y_test_pred = np.argmax(y_test_pred_prob, axis=1)
         
            if y_test_pred==1:
                print("DDoS")
                pass
            # else:
            #     try:
            #         with open("test.csv", 'w') as csvfile:
            #             writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            #             writer.writeheader()
            #             for data in dict:
            #                 writer.writerow(data)
                
            #     except IOError:
                    # print("I/O error")
                            


        
    def garbage_collect(self, latest_time) -> None:
        # TODO: Garbage Collection / Feature Extraction should have a separate thread
        model= load_model('AllData')
        # print(model.summary())

        data_queue = Queue(maxsize=0)
        datas_queue = Queue(maxsize=0)


        if not self.url_model:
            print("Garbage Collection Began. Flows = {}".format(len(self.flows)))
        keys = list(self.flows.keys())
        
        for k in keys:
            flow = self.flows.get(k)

            if (
                latest_time is None
                or latest_time - flow.latest_timestamp > EXPIRED_UPDATE
                or flow.duration > 90
            ):
                data = flow.get_data()
                # print(data)

                # POST Request to Model API
                if self.url_model:
                    payload = {
                        "columns": list(data.keys()),
                        "data": [list(data.values())],
                    }
                    post = requests.post(
                        self.url_model,
                        json=payload,
                        headers={
                            "Content-Type": "application/json; format=pandas-split"
                        },
                    )
                    resp = post.json()
                    result = resp["result"].pop
                    if result == 0:
                        result_print = "Benign"
                    else:
                        result_print = "Malicious"

                    print(
                        "{: <15}:{: <6} -> {: <15}:{: <6} \t {} (~{:.2f}%)".format(
                            resp["src_ip"],
                            resp["src_port"],
                            resp["dst_ip"],
                            resp["dst_port"],
                            result_print,
                            resp["probability"].pop()[result] * 100,
                        )
                    )
                

              
                # print(data)

                data_queue.put(data)
                if (data_queue.qsize()>=5):
                    timer = threading.Timer(2.0, self.get_label,args=(model,data_queue,))
                    timer.start()                    
                else:
                    print("size of queue = ",data_queue.qsize())
            

                
                
                
                # print(data.values())
                # if self.csv_line == 0:
                #     self.csv_writer.writerow(data.keys())

                # self.csv_writer.writerow(data.values())
                # self.csv_line += 1

                del self.flows[k]
        if not self.url_model:
            print("Garbage Collection Finished. Flows = {}".format(len(self.flows)))


def generate_session_class(output_mode, output_file, url_model):
    print("start")
    return type(
        "NewFlowSession",
        (FlowSession,),
        {
            "output_mode": output_mode,
            "output_file": output_file,
            "url_model": url_model,
        },
    )
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam',metrics=['accuracy'])
    return model






