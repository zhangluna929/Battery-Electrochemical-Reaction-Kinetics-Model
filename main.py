import numpy as np


class ElectrochemicalReactionModel:
    def __init__(self, capacity, resistance, voltage, temperature=25, max_soc=1.0, min_soc=0.0,
                 nominal_voltage=3.7, reaction_rate_constant=1e-10, ion_diffusivity=1e-14,
                 electrode_area=0.02, electrolyte_conductivity=1.0, cycle_life=1000):
        """
        初始化电池电化学反应动力学模型，包含温度效应、离子迁移、反应速率等多个因素(good luck!)。

        参数:
        capacity (float): 电池容量 (Ah)
        resistance (float): 电池内部电阻 (ohm)
        voltage (float): 电池额定电压 (V)
        temperature (float): 电池初始温度 (°C)
        max_soc (float): 最大SOC（充电状态）
        min_soc (float): 最小SOC（充电状态）
        nominal_voltage (float): 电池额定电压 (V)
        reaction_rate_constant (float): 电化学反应速率常数
        ion_diffusivity (float): 离子扩散系数
        electrode_area (float): 电极表面积 (m²)
        electrolyte_conductivity (float): 电解质的导电率 (S/m)
        cycle_life (int): 电池的最大循环寿命
        """
        self.capacity = capacity  # 电池容量 (Ah)
        self.resistance = resistance  # 内部电阻 (ohm)
        self.voltage = voltage  # 电池电压 (V)
        self.nominal_voltage = nominal_voltage  # 电池额定电压 (V)
        self.temperature = temperature  # 电池温度 (°C)
        self.state_of_charge = 1.0  # 电池初始SOC
        self.max_soc = max_soc  # 最大SOC（充电状态）
        self.min_soc = min_soc  # 最小SOC（充电状态）
        self.reaction_rate_constant = reaction_rate_constant  # 电化学反应速率常数
        self.ion_diffusivity = ion_diffusivity  # 离子扩散系数
        self.electrode_area = electrode_area  # 电极表面积 (m²)
        self.electrolyte_conductivity = electrolyte_conductivity  # 电解质导电率 (S/m)
        self.cycle_life = cycle_life  # 电池的最大循环寿命
        self.cycle_count = 0  # 当前循环次数
        self.degradation_factor = 1.0  # 电池退化因子
        self.temperature_effect_factor = 1.0  # 温度效应因子
        self.ion_concentration = 1.0  # 离子浓度（可根据不同的电池设计调整）

    def calculate_reaction_rate(self, current_density):
        """
        计算电池的电化学反应速率，考虑电流密度、温度、SOC等因素。

        参数:
        current_density (float): 电流密度 (A/m²)

        返回:
        float: 电化学反应速率 (mol/s·m²)
        """
        # 电化学反应速率的经典模型：r = k * j
        # k: 反应速率常数，j: 电流密度，考虑温度效应
        temperature_factor = np.exp(-self.temperature / 300)  # 温度对反应速率的影响，假设为Arrhenius关系
        reaction_rate = self.reaction_rate_constant * current_density * temperature_factor
        return reaction_rate

    def update_state_of_charge(self, current, time_step):
        """
        更新电池的充电状态 (SOC)，通过电流和时间步长计算。

        参数:
        current (float): 电池电流 (A)
        time_step (float): 每次更新的时间步长 (秒)

        返回:
        float: 更新后的充电状态 (SOC)
        """
        delta_capacity = current * time_step  # 电池容量变化 (A·s)
        self.state_of_charge += delta_capacity / self.capacity

        # 限制SOC在0和1之间
        self.state_of_charge = np.clip(self.state_of_charge, self.min_soc, self.max_soc)
        return self.state_of_charge

    def calculate_voltage(self):
        """
        计算电池电压，考虑SOC、温度和电池内阻对电压的影响。

        返回:
        float: 当前电池电压 (V)
        """
        voltage_drop = self.resistance * (self.voltage - self.nominal_voltage)  # 电池内部电阻的电压降
        voltage = self.nominal_voltage * self.state_of_charge - voltage_drop
        return voltage

    def calculate_ion_diffusion(self, concentration_gradient):
        """
        计算离子扩散的影响，模拟离子在电池中的迁移效应。

        参数:
        concentration_gradient (float): 离子浓度梯度 (mol/m³)

        返回:
        float: 离子扩散引起的电压变化 (V)
        """
        ion_diffusion_effect = self.ion_diffusivity * concentration_gradient  # 离子扩散效应
        return ion_diffusion_effect

    def calculate_temperature_effect(self):
        """
        计算温度对电池反应速率、电池内阻的影响。

        返回:
        float: 温度对电池性能的影响
        """
        # 假设温度每升高10°C，电池内阻增加2%
        resistance_increase = 0.02 * (self.temperature - 25)  # 每升高10°C，内阻增加2%
        self.resistance += self.resistance * resistance_increase
        self.temperature_effect_factor = np.exp(-self.temperature / 300)  # 温度效应对反应速率的影响
        return self.temperature_effect_factor

    def simulate_degradation(self):
        """
        模拟电池退化，随着使用次数的增加，电池性能逐渐下降。

        返回:
        float: 退化因子的变化
        """
        if self.cycle_count > self.cycle_life:
            self.degradation_factor -= 0.01  # 退化因子降低
            self.degradation_factor = np.clip(self.degradation_factor, 0.1, 1.0)  # 退化因子最低为0.1

    def check_battery_status(self):
        """
        检查电池状态，是否存在过热、SOC过低等异常情况。

        返回:
        str: 电池状态（"正常"、"电池过热"、"SOC过低"）
        """
        if self.temperature > 60:
            return "电池过热"
        elif self.state_of_charge <= self.min_soc:
            return "SOC过低"
        elif self.state_of_charge >= self.max_soc:
            return "SOC过高"
        else:
            return "正常"

    def simulate(self, current, time_steps=10):
        """
        仿真电池在多个时间步长中的行为。

        参数:
        current (float): 电池电流 (A)
        time_steps (int): 仿真时长（单位：秒）
        """
        for t in range(time_steps):
            # 更新SOC
            self.update_state_of_charge(current, time_step=1)

            # 计算温度效应
            self.calculate_temperature_effect()

            # 计算电池电化学反应速率
            current_density = current / self.electrode_area  # 计算电流密度
            reaction_rate = self.calculate_reaction_rate(current_density)

            # 计算电池电压
            voltage = self.calculate_voltage()

            # 模拟退化
            self.simulate_degradation()

            # 检查电池状态
            status = self.check_battery_status()

            # 打印当前状态
            print(f"时间: {t}s, 电压: {voltage}V, SOC: {self.state_of_charge:.3f}, "
                  f"反应速率: {reaction_rate:.4e}, 温度: {self.temperature:.2f}°C, "
                  f"退化因子: {self.degradation_factor:.2f}, 状态: {status}")


# 测试电池电化学反应动力学模型
if __name__ == "__main__":
    # 创建电池对象，容量50Ah，内阻0.1Ω，额定电压3.7V，环境温度25°C
    battery = ElectrochemicalReactionModel(capacity=50, resistance=0.1, voltage=3.7)

    # 模拟电池放电过程
    print("开始仿真：")
    battery.simulate(current=-5, time_steps=10)
