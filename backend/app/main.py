from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from app.services.backtesting_service import BacktestingService
from app.services.strategy_storage import StrategyStorage
from app.auth.auth import create_access_token, get_current_active_user, User
from datetime import timedelta

app = FastAPI()

backtesting_service = BacktestingService()
strategy_storage = StrategyStorage()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/optimize")
async def optimize_strategy(strategy_name: str, param_grid: dict, current_user: User = Depends(get_current_active_user)):
    best_sharpe, best_params = backtesting_service.optimize_strategy(strategy_name, param_grid)
    strategy_storage.save_strategy(strategy_name, best_params)
    return {"sharpe_ratio": best_sharpe, "best_params": best_params}

@app.get("/strategies")
async def list_strategies(current_user: User = Depends(get_current_active_user)):
    return strategy_storage.list_strategies()

@app.get("/strategy/{strategy_name}")
async def get_strategy(strategy_name: str, current_user: User = Depends(get_current_active_user)):
    strategy = strategy_storage.load_strategy(strategy_name)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy

@app.post("/backtest")
async def run_backtest(strategy_name: str, params: dict, start_date: str, end_date: str, current_user: User = Depends(get_current_active_user)):
    backtesting_service.load_data("AAPL", "1d", start_date, end_date)  # Замените на реальную загрузку данных
    strategy = strategy_storage.load_strategy(strategy_name)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    sharpe_ratio, returns = backtesting_service.run_backtest(strategy, params)
    return {"sharpe_ratio": sharpe_ratio, "returns": returns.tolist()}
