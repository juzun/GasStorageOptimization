import streamlit as st
from gas_tank.gs_optim import GasStorage
import datetime as dt

def main(state=None):
    st.title('Nazdar')

    storage_rwe = GasStorage('RWE', dt.date(2023,4,1), dt.date(2024,3,31))
    
    st.write(f'ID ásobníku: {storage_rwe.id}')



if __name__ == "__main__":
    main()