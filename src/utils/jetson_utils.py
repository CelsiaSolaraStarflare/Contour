"""
Jetson Utilities Module
Optimization utilities for NVIDIA Jetson Nano performance
"""

import subprocess
import logging
import psutil
import time
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)


def setup_jetson():
    """Setup Jetson Nano for optimal performance"""
    try:
        # Set power mode to MAXN (10W)
        set_power_mode(0)
        
        # Set GPU and CPU frequencies
        set_gpu_frequency(921600000)  # 921.6 MHz
        set_cpu_frequency(1479000000)  # 1.479 GHz
        
        # Enable jetson_clocks
        enable_jetson_clocks()
        
        # Set memory allocation
        set_memory_allocation()
        
        logger.info("Jetson Nano optimized for performance")
        
    except Exception as e:
        logger.warning(f"Could not optimize Jetson Nano: {e}")


def set_power_mode(mode: int):
    """Set Jetson Nano power mode"""
    try:
        # Mode 0: MAXN (10W)
        # Mode 1: 5W
        # Mode 2: 5W (alternative)
        subprocess.run(['sudo', 'nvpmodel', '-m', str(mode)], check=True)
        logger.info(f"Power mode set to {mode}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to set power mode: {e}")


def enable_jetson_clocks():
    """Enable jetson_clocks for maximum performance"""
    try:
        subprocess.run(['sudo', 'jetson_clocks'], check=True)
        logger.info("Jetson clocks enabled")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to enable jetson_clocks: {e}")


def set_gpu_frequency(freq: int):
    """Set GPU frequency in Hz"""
    try:
        # Set GPU frequency
        with open('/sys/kernel/debug/clock/gpcclk/rate', 'w') as f:
            f.write(str(freq))
        logger.info(f"GPU frequency set to {freq} Hz")
    except Exception as e:
        logger.warning(f"Failed to set GPU frequency: {e}")


def set_cpu_frequency(freq: int):
    """Set CPU frequency in Hz"""
    try:
        # Set CPU frequency for all cores
        for i in range(4):  # Jetson Nano has 4 CPU cores
            with open(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_setspeed', 'w') as f:
                f.write(str(freq))
        logger.info(f"CPU frequency set to {freq} Hz")
    except Exception as e:
        logger.warning(f"Failed to set CPU frequency: {e}")


def set_memory_allocation():
    """Set memory allocation for GPU"""
    try:
        # Set GPU memory allocation (in MB)
        gpu_memory = 2048  # 2GB for GPU
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-c', '0'], check=True)
        logger.info(f"GPU memory allocation set to {gpu_memory} MB")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to set memory allocation: {e}")


def get_system_info() -> Dict:
    """Get Jetson Nano system information"""
    info = {}
    
    try:
        # CPU info
        info['cpu_count'] = psutil.cpu_count()
        info['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        
        # Memory info
        memory = psutil.virtual_memory()
        info['memory_total'] = memory.total
        info['memory_available'] = memory.available
        info['memory_percent'] = memory.percent
        
        # GPU info
        gpu_info = get_gpu_info()
        info.update(gpu_info)
        
        # Temperature
        info['temperature'] = get_temperature()
        
        # Power consumption
        info['power_consumption'] = get_power_consumption()
        
    except Exception as e:
        logger.warning(f"Error getting system info: {e}")
    
    return info


def get_gpu_info() -> Dict:
    """Get GPU information"""
    info = {}
    
    try:
        # Run nvidia-smi to get GPU info
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        # Parse output
        lines = result.stdout.strip().split('\n')
        if lines:
            values = lines[0].split(', ')
            if len(values) >= 5:
                info['gpu_memory_total'] = int(values[0])
                info['gpu_memory_used'] = int(values[1])
                info['gpu_memory_free'] = int(values[2])
                info['gpu_utilization'] = int(values[3])
                info['gpu_temperature'] = int(values[4])
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to get GPU info: {e}")
    
    return info


def get_temperature() -> Optional[float]:
    """Get system temperature"""
    try:
        # Read thermal zone temperature
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000.0  # Convert from millidegrees
        return temp
    except Exception as e:
        logger.warning(f"Failed to get temperature: {e}")
        return None


def get_power_consumption() -> Optional[float]:
    """Get power consumption in watts"""
    try:
        # Read power consumption from INA3221
        with open('/sys/bus/i2c/devices/1-0040/iio_device/in_power0_input', 'r') as f:
            power = int(f.read().strip()) / 1000.0  # Convert from milliwatts
        return power
    except Exception as e:
        logger.warning(f"Failed to get power consumption: {e}")
        return None


def monitor_performance(duration: int = 60, interval: int = 1):
    """Monitor system performance for specified duration"""
    logger.info(f"Starting performance monitoring for {duration} seconds")
    
    start_time = time.time()
    data_points = []
    
    while time.time() - start_time < duration:
        try:
            # Get system info
            info = get_system_info()
            info['timestamp'] = time.time()
            data_points.append(info)
            
            # Log current status
            logger.info(f"CPU: {info.get('cpu_freq', 0):.0f} MHz, "
                       f"GPU: {info.get('gpu_utilization', 0)}%, "
                       f"Memory: {info.get('memory_percent', 0):.1f}%, "
                       f"Temp: {info.get('temperature', 0):.1f}°C")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
            break
    
    logger.info("Performance monitoring completed")
    return data_points


def optimize_for_inference():
    """Optimize Jetson Nano specifically for inference"""
    try:
        # Set power mode to MAXN
        set_power_mode(0)
        
        # Enable jetson_clocks
        enable_jetson_clocks()
        
        # Set GPU memory allocation
        set_memory_allocation()
        
        # Set CPU governor to performance
        set_cpu_governor('performance')
        
        # Disable unnecessary services
        disable_services()
        
        logger.info("Jetson Nano optimized for inference")
        
    except Exception as e:
        logger.warning(f"Could not optimize for inference: {e}")


def set_cpu_governor(governor: str):
    """Set CPU governor"""
    try:
        for i in range(4):  # 4 CPU cores
            with open(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor', 'w') as f:
                f.write(governor)
        logger.info(f"CPU governor set to {governor}")
    except Exception as e:
        logger.warning(f"Failed to set CPU governor: {e}")


def disable_services():
    """Disable unnecessary services to free up resources"""
    services_to_disable = [
        'bluetooth',
        'wifi-powersave',
        'snapd'
    ]
    
    for service in services_to_disable:
        try:
            subprocess.run(['sudo', 'systemctl', 'stop', service], 
                         capture_output=True, check=False)
            subprocess.run(['sudo', 'systemctl', 'disable', service], 
                         capture_output=True, check=False)
        except Exception as e:
            logger.debug(f"Could not disable {service}: {e}")


def enable_services():
    """Re-enable services"""
    services_to_enable = [
        'bluetooth',
        'wifi-powersave',
        'snapd'
    ]
    
    for service in services_to_enable:
        try:
            subprocess.run(['sudo', 'systemctl', 'enable', service], 
                         capture_output=True, check=False)
            subprocess.run(['sudo', 'systemctl', 'start', service], 
                         capture_output=True, check=False)
        except Exception as e:
            logger.debug(f"Could not enable {service}: {e}")


def check_thermal_throttling() -> bool:
    """Check if thermal throttling is active"""
    try:
        # Check thermal zone status
        with open('/sys/class/thermal/thermal_zone0/cdev0/cur_state', 'r') as f:
            state = int(f.read().strip())
        
        # Check temperature
        temp = get_temperature()
        
        # Thermal throttling if temperature > 80°C or cooling state > 0
        return temp > 80.0 or state > 0
        
    except Exception as e:
        logger.warning(f"Could not check thermal throttling: {e}")
        return False


def get_optimal_batch_size(model_size_mb: float) -> int:
    """Calculate optimal batch size based on model size and available memory"""
    try:
        # Get available GPU memory
        gpu_info = get_gpu_info()
        available_memory = gpu_info.get('gpu_memory_free', 2048)  # MB
        
        # Reserve some memory for system
        usable_memory = available_memory * 0.8
        
        # Calculate batch size (rough estimate)
        # Assume 2x model size for intermediate activations
        memory_per_sample = model_size_mb * 2
        
        optimal_batch_size = max(1, int(usable_memory / memory_per_sample))
        
        logger.info(f"Optimal batch size: {optimal_batch_size} "
                   f"(GPU memory: {available_memory} MB, model size: {model_size_mb} MB)")
        
        return optimal_batch_size
        
    except Exception as e:
        logger.warning(f"Could not calculate optimal batch size: {e}")
        return 1


def cleanup_resources():
    """Clean up system resources"""
    try:
        # Reset power mode to default
        set_power_mode(1)
        
        # Reset CPU governor
        set_cpu_governor('ondemand')
        
        # Re-enable services
        enable_services()
        
        logger.info("System resources cleaned up")
        
    except Exception as e:
        logger.warning(f"Could not cleanup resources: {e}")


class JetsonMonitor:
    """Real-time Jetson Nano monitoring"""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize monitor"""
        self.log_file = log_file
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.is_monitoring = True
        logger.info("Real-time monitoring started")
        
        # In a real implementation, you would start a background thread here
        # For now, we'll just log that monitoring is active
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        logger.info("Real-time monitoring stopped")
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return get_system_info()
    
    def log_status(self):
        """Log current status to file"""
        if self.log_file:
            status = self.get_status()
            with open(self.log_file, 'a') as f:
                f.write(f"{time.time()},{status}\n") 