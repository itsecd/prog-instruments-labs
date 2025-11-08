import math
from model_objects import Discount, Offer, SpecialOfferType
from receipt import Receipt


class Teller:

    def __init__(self, catalog):
        self.catalog = catalog
        self.offers = {}

    def add_special_offer(self, offer_type, product, argument):
        self.offers[product] = Offer(offer_type, product, argument)

    def checks_out_articles_from(self, the_cart):
        receipt = Receipt()
        product_quantities = the_cart.items
        for pq in product_quantities:
            p = pq.product
            quantity = pq.quantity
            unit_price = self.catalog.unit_price(p)
            price = quantity * unit_price
            receipt.add_product(p, quantity, unit_price, price)

        self.handle_offers(receipt, the_cart)

        return receipt
      
    
    def handle_special_offers(self, offer, quantity, unit_price, p)->int:
        if offer.offer_type == SpecialOfferType.THREE_FOR_TWO and quantity<=2:
            return None
        if offer.offer_type == SpecialOfferType.TWO_FOR_AMOUNT and quantity<2:
            return None
        if offer.offer_type == SpecialOfferType.FIVE_FOR_AMOUNT and quantity<5:
            return None

        if offer.offer_type == SpecialOfferType.THREE_FOR_TWO:
            x = 3
            number_of_x = quantity//x
            discount_amount = quantity * unit_price - ((number_of_x * 2 * unit_price) + quantity % 3 * unit_price)
            discount = Discount(p, "3 for 2", -discount_amount)
            
        elif offer.offer_type == SpecialOfferType.TWO_FOR_AMOUNT:
            x = 2
            number_of_x = quantity//x
            total = offer.argument * (quantity // x) + quantity % 2 * unit_price
            discount_n = unit_price * quantity - total
            discount = Discount(p, "2 for " + str(offer.argument), -discount_n)
                
        elif offer.offer_type == SpecialOfferType.FIVE_FOR_AMOUNT:
            x = 5
            number_of_x = quantity//x
            discount_total = unit_price * quantity - (
                offer.argument * number_of_x + quantity % 5 * unit_price)
            discount = Discount(p, str(x) + " for " + str(offer.argument), -discount_total)

        
        elif offer.offer_type == SpecialOfferType.TEN_PERCENT_DISCOUNT:
            discount = Discount(p, str(offer.argument) + "% off",
                                -quantity * unit_price * offer.argument / 100.0)
            
        return discount
    

    def handle_offers(self, receipt, cart):
        for p in cart.product_quantities.keys():
            quantity = cart.product_quantities[p]
            if p in self.offers.keys():
                offer = self.offers[p]
                unit_price = self.catalog.unit_price(p)
                quantity_as_int = int(quantity)
                
                discount = self.handle_special_offers(offer, quantity_as_int, unit_price, p)
                if discount:
                    receipt.add_discount(discount)