    # def log(self, line, time_head=True):
    #     if not self.core.auto_log:
    #         return
    #     if self.under_test:
    #         line = f'testing# {line}'
    #     if time_head:
    #         line = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] {line}'
    #     print(line)
    #     fp = self.core.slave_log_path
    #     if fp is None:
    #         fp = f'{self.root}/slave.log'
    #     with open(fp, 'a') as f:
    #         f.write(f'{line}\n')

    # def netdata_log(self, metrics_dict):
    #     if not self.core.auto_log:
    #         return
    #     if self.core.netdata_log_path is not None:
    #         self.netdata_state.update(metrics_dict)
    #         with open(self.core.netdata_log_path, "w")as f:
    #             json.dump(self.netdata_state, f)

    # def hyper_log(self, msg, time_head=True, hash_head=True, iter_head=True):
    #     if not self.core.auto_log:
    #         return
    #     if type(msg) is dict:
    #         for k, v in msg.items():
    #             self.hyper_log(f'{k}: {v}', time_head=time_head, hash_head=hash_head, iter_head=iter_head)
    #         return self
    #     if iter_head:
    #         msg = f'[{self.status.epoch}, {self.status.iter:5d}/{len(self.trainloader)}] {msg}'
    #     if hash_head:
    #         msg = f'<{self.status.hash}>\t{msg}'
    #     self.log(msg, time_head=time_head)