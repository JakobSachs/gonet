import multiprocessing as mp
from model import ValueNet
from self_play import self_play_worker
from train import trainer
from ws_server import start_ws
from config import cfg  # your simple config holder

def main():
    mp.set_start_method("spawn")
    # 1) shared net
    net = ValueNet(cfg.board_size)
    net.share_memory()

    # 2) queues
    train_q = mp.Queue(maxsize=cfg.buf_size*2)
    event_q = mp.Queue(maxsize=cfg.buf_size*4)

    # 3) spawn trainer
    p_tr = mp.Process(target=trainer, args=(net, train_q, cfg))
    p_tr.start()

    # 4) spawn workers
    procs = []
    for wid in range(cfg.num_workers):
        p = mp.Process(target=self_play_worker,
                       args=(wid, net, train_q, event_q, cfg))
        p.start()
        procs.append(p)

    # 5) run websocket server in main
    start_ws(cfg.ws_host, cfg.ws_port, event_q)

    p_tr.join()
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
