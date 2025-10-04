from config import START_MESSAGE, BLACKJACK_STR
from Deck import Deck
from Player import Player, Dealer


class BlackJack2:

    def __init__(self):
        self.player = Player("Player", 100)
        self.dealer = Dealer("Dealer", 10000)
        self.game_num = 0

    def start_round(self, bet: int):
        self.game_num += 1
        self.player.reset_hand()
        self.dealer.reset_hand()
        self.deck = Deck()
        self.deck.shuffle()

        bet_amount = self.player.make_bet(bet)

        for _ in range(2):
            self.player.add_card(self.deck.deal_one())
            self.dealer.add_card(self.deck.deal_one())

        return bet_amount








def check_ace(card):
    """
    function to check for ace and adjust its value according to the user
    """
    if card.rank == "Ace":
        while True:
            try:
                ace_val = int(input("\nWhat value do you want to consider for Ace (1/11)? :"))
                match ace_val:
                    case 1:
                        values["Ace"] = 1
                        break
                    case 11:
                        values["Ace"] = 11
                        break
            except Exception as exp:
                print("Input a integer: 1 or 11")


class BlackJack:
    def __init__(self):
        self._chips = 100
        self._game_num = 0
        self._game_on = True

    @staticmethod
    def __print_start_message():
        print("\n" * 100)

        print(BLACKJACK_STR)

        print(START_MESSAGE)

    def __make_bet(self):
        while True:
            bet = int(input("\nEnter the amount of chips you want to bet:"))
            if bet > self._chips:
                print("You dont have enough chips.")
                print("Enter a valid amount. \n")
            elif bet <= 0:  # To prevent betting a negative value
                print("Invalid Bet")
            else:
                self._chips -= bet
                return bet

    def __start_deal(self, new_deck: Deck):
        player_table_cards = []  # cards on table will be replaced each round
        dealer_table_cards = []
        # using list comprehension to distribute 2 cards
        [player_table_cards.append(new_deck.deal_one()) for i in range(2)]
        # to both user and dealer(computer)
        [dealer_table_cards.append(new_deck.deal_one()) for i in range(2)]

        print(f"\nPlayer cards are {player_table_cards[0]} and {player_table_cards[1]}")
        print(f"Dealer cards are {dealer_table_cards[0]} and Hidden.")

        # checking both the cards given to the user for being ace
        check_ace(player_table_cards[0])
        check_ace(player_table_cards[1])

        return player_table_cards, dealer_table_cards

    @staticmethod
    def __compute_sum(table_cards: list):
        sum_cards_val = 0
        for i in table_cards:
            sum_cards_val += i.value
        return sum_cards_val

    def __print_hand(self, table_cards: list, hand_owner: str):
        print(F"\n{hand_owner}'s hand :")
        # using list comprehension to print cards on table
        [print(i) for i in table_cards]
        print()
    def __hit_or_stand(self, player_table_cards: list, new_deck: Deck) -> int:
        player_cards_val = 0
        while player_cards_val < 21:
            hit_or_stand = input("Do you want to hit or stand? :").lower()
            match hit_or_stand:
                case "hit":
                    player_table_cards.append(new_deck.deal_one())
                    check_ace(player_table_cards[-1])

                    print(f"\nThe player hits card : {player_table_cards[-1]}")

                    self.__print_hand(player_table_cards, "Player")

                    player_cards_val = self.__compute_sum(player_table_cards)
                    continue

                case "stand":
                    player_cards_val = self.__compute_sum(player_table_cards)

                    print("\nPlayer has decided to stand.")

                    self.__print_hand(player_table_cards, "Player")
                    return player_cards_val

                case _:
                    print("Enter a valid option. \n")
                    continue
        return player_cards_val

    def __dealer_logic(self, dealer_table_cards: list, new_deck: Deck):
        no_of_hits = 0
        dealer_cards_val = 0
        while dealer_cards_val < 21:

            dealer_cards_val = self.__compute_sum(dealer_table_cards)

            if dealer_cards_val < 17:
                no_of_hits += 1
                dealer_table_cards.append(new_deck.deal_one())
                continue

            elif 17 <= dealer_cards_val < 21:
                print(f"The Dealer has hit {no_of_hits} times.")
                self.__print_hand(dealer_table_cards, "Dealer")
                return dealer_cards_val, no_of_hits

            elif dealer_cards_val == 21:
                print(f"The Dealer has hit {no_of_hits} times.")
                print("The Dealer got a blackjack!")
                self.__print_hand(dealer_table_cards, "Dealer")
                return dealer_cards_val, no_of_hits
        return dealer_cards_val, no_of_hits

    def __summing_up(self, player_cards_val: int, dealer_cards_val: int, bet: int, no_of_hits: int, dealer_table_cards: list):
        if dealer_cards_val > 21:
            # checking if player has also busted or not. If player busts , dealer's bust doesn't count.
            if not (player_cards_val > 21):
                print(f"The Dealer has hit {no_of_hits} times.")
                print("The Dealer busted!")
                self.__print_hand(dealer_table_cards, "Dealer")


            # checking for busts first
        if player_cards_val > 21:
            print(
                "\nSince the Player busted , the round is lost.\nPlayer lost the bet money"
            )

        elif dealer_cards_val > 21:
            print(
                "\n Since the Dealer busted , Player won the round! \nPlayer got twice the money bet."
            )
            self._chips += bet * 2

            # checking for player's blackjack then
        elif player_cards_val == 21:
            print("\nPlayer won with a blackjack! \nPlayer got thrice the money bet.")
            self._chips += bet * 3

            # checking whose value is closer to 21
        elif 21 - player_cards_val > 21 - dealer_cards_val:
            print("\nDealer won the round. \nPlayer lost the bet money")

        elif 21 - dealer_cards_val > 21 - player_cards_val:
            print("\nPlayer won the round. \nPlayer got twice the money bet.")
            self._chips += bet * 2

            # last situation can only be a tie
        else:
            print("\nIt's a tie. \nBet money was returned.")
            self._chips += bet

    def __restart_game(self):
        match self._chips:
            case 0:
                print("\nYou are out of chips , Game over.")
                self._game_on = False

            case _:
                cont = input("Do you want to continue? (y/n) :")
                check = cont.upper()  ###So a capital or lowercase value can be entered
                match check:
                    case "Y":
                        print("\n" * 100)

                        print(BLACKJACK_STR)

                    case _:
                        print(f"\nTotal amount of chips left with the player = {self._chips}")
                        print(input("Press Enter to exit the terminal..."))
                        self._game_on = False

    def start(self):
            BlackJack.__print_start_message()

            while self._game_on:
                try:
                    new_deck = Deck()  # new deck will be created and shuffled each round
                    new_deck.shuffle()

                    self._game_num += 1
                    print(f"\nGame Round number : {self._game_num}")
                    print(f"Chips remaining = {self._chips}")

                    bet = self.__make_bet()

                    player_table_cards, dealer_table_cards = self.__start_deal(new_deck)

                    player_cards_val = self.__hit_or_stand(player_table_cards, new_deck)



                    # variable that stores how many times dealer hits before its cards value is more than equal to 17
                    dealer_cards_val, no_of_hits = self.__dealer_logic(dealer_table_cards, new_deck)

                    self.__summing_up(player_cards_val, dealer_cards_val, bet, no_of_hits, dealer_table_cards)

                    self.__restart_game()
                except Exception as error:
                    print(f"Following error occurred : {error} \nPlease try again.")
                    self._game_num -= 1  # round with error won't be counted
                    continue




