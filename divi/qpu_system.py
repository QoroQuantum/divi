from typing import List


class QPU:
    def __init__(self, nickname: str, q_bits: int, status: str, system_kind: str):
        self.nickname = nickname
        self.q_bits = q_bits
        self.status = status
        self.system_kind = system_kind

    def __repr__(self):
        return (
            f"QPU(nickname='{self.nickname}', q_bits={self.q_bits}, "
            f"status='{self.status}', system_kind='{self.system_kind}')"
        )


class QPUSystem:
    def __init__(self, name: str, qpus: List[QPU], access_level: str):
        self.name = name
        self.qpus = qpus
        self.access_level = access_level

    def __repr__(self):
        return (
            f"QPUSystem(name='{self.name}', access_level='{self.access_level}', "
            f"qpus={self.qpus})"
        )


def parse_qpu_systems(json_data: list) -> List[QPUSystem]:
    systems = []
    for system_data in json_data:
        qpus = [QPU(**qpu_data) for qpu_data in system_data.get("qpus", [])]
        system = QPUSystem(
            name=system_data["name"],
            qpus=qpus,
            access_level=system_data["access_level"],
        )
        systems.append(system)
    return systems
