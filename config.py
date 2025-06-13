import os
from typing import Optional

class Config:
    """Configuration class for VoluGuard API"""
    
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://eurepsbikwxwmgpgzvzn.supabase.co")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV1cmVwc2Jpa3d4d21ncGd6dnpuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0MTYzNzQsImV4cCI6MjA2NDk5MjM3NH0.r3soCQV8nkbvc8RzFoLNGxK9MqQUOEIQUAWubAzAIkA")
    
    # Server Configuration
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Trading Configuration
    TOTAL_FUNDS: int = int(os.getenv("TOTAL_FUNDS", 2000000))
    DAILY_RISK_LIMIT_PCT: float = float(os.getenv("DAILY_RISK_LIMIT_PCT", 0.02))
    WEEKLY_RISK_LIMIT_PCT: float = float(os.getenv("WEEKLY_RISK_LIMIT_PCT", 0.03))
    LOT_SIZE: int = int(os.getenv("LOT_SIZE", 75))
    
    # API URLs
    UPSTOX_BASE_URL: str = "https://api.upstox.com/v2"
    INSTRUMENT_KEY: str = "NSE_INDEX|Nifty 50"
    EVENT_URL: str = "https://raw.githubusercontent.com/shritish20/VolGuard/main/upcoming_events.csv"
    IVP_URL: str = "https://raw.githubusercontent.com/shritish20/VolGuard/main/ivp.csv"
    NIFTY_URL: str = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
    MODEL_URL: str = "https://raw.githubusercontent.com/shritish20/VolGuard-Pro/main/xgb_vol_model_v2.pkl"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_vars = [
            "SUPABASE_URL",
            "SUPABASE_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
    
    @classmethod
    def get_supabase_headers(cls) -> dict:
        """Get Supabase headers for API requests"""
        return {
            "Authorization": f"Bearer {cls.SUPABASE_KEY}",
            "apikey": cls.SUPABASE_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
    
    @classmethod
    def get_risk_config(cls) -> dict:
        """Get risk configuration for different strategies"""
        return {
            "Iron Fly": {"capital_pct": 0.30, "risk_per_trade_pct": 0.01},
            "Iron Condor": {"capital_pct": 0.25, "risk_per_trade_pct": 0.015},
            "Jade Lizard": {"capital_pct": 0.20, "risk_per_trade_pct": 0.01},
            "Straddle": {"capital_pct": 0.15, "risk_per_trade_pct": 0.02},
            "Calendar Spread": {"capital_pct": 0.10, "risk_per_trade_pct": 0.01},
            "Bull Put Spread": {"capital_pct": 0.15, "risk_per_trade_pct": 0.01},
            "Wide Strangle": {"capital_pct": 0.10, "risk_per_trade_pct": 0.015},
            "ATM Strangle": {"capital_pct": 0.10, "risk_per_trade_pct": 0.015}
        }

# Initialize configuration
config = Config()

# Validate configuration on import
try:
    config.validate_config()
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please check your environment variables and .env file")

