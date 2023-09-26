# 定义奶茶的基本价格
base_price = {
    "中杯": 6,
    "大杯": 8,
    "超大杯": 10
}

tea_price = {
    "乌龙茶": 2,
    "绿茶": 3,
    "红茶": 4
}

toppings_price = {
    "珍珠": 1,
    "椰果": 2,
    "奶盖": 3,
    "不加": 0
}

# 奶茶定制装饰器
def customize_milk_tea(price = base_price):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cup = input("请选择杯型（中杯/大杯/超大杯）: ")
            tea = input("请选择茶底（乌龙茶/绿茶/红茶）: ")
            toppings = input("请选择加料（珍珠/椰果/奶盖/不加，多选请用逗号分隔）: ")

            # 计算奶茶价格
            total_price = price[cup] + tea_price[tea]
            topping_list = toppings.split(',')
            for topping in topping_list:
                total_price += toppings_price[topping]

            # 输出奶茶信息和总价
            print(f"您点的奶茶：杯型-{cup}，茶底-{tea}，加料-{toppings}")
            print(f"总价：{total_price}元")

        return wrapper

    return decorator

# 定义奶茶定制函数
@customize_milk_tea()
def order_milk_tea():
    pass

# 主程序
if __name__ == "__main__":
    order_milk_tea()
