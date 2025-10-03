from config import values, START_MESSAGE, BLACKJACK_STR
from Deck import Deck




def check_ace(card):
    """
    function to check for ace and adjust its value according to the user
    """
    if card.rank == "Ace":
        while True:
            ace_val = int(
                input("\nWhat value do you want to consider for Ace (1/11)? :")
            )

            if ace_val == 1:
                values["Ace"] = 1

                break
            elif ace_val == 11:
                values["Ace"] = 11

                break
            else:
                print("choose valid value.")
                continue


def main():
    chips = 100
    game_num = 0
    game_on = True

    print("\n" * 100)  # clears up the terminal for a cleaner look

    print(BLACKJACK_STR)

    print(START_MESSAGE)

    while game_on:
        try:
            new_deck = Deck()  # new deck will be created and shuffled each round
            new_deck.shuffle()

            game_num += 1
            print(f"\nGame Round number : {game_num}")
            print(f"Chips remaining = {chips}")

            while True:
                bet = int(input("\nEnter the amount of chips you want to bet:"))
                if bet > chips:
                    print("You dont have enough chips.")
                    print("Enter a valid amount. \n")
                elif bet <= 0:  # To prevent betting a negative value
                    print("Invalid Bet")
                else:
                    chips -= bet
                    break

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

            while True:
                hit_or_stand = input("Do you want to hit or stand? :").lower()

                if hit_or_stand == "hit":
                    player_table_cards.append(new_deck.deal_one())
                    check_ace(player_table_cards[-1])

                    print(f"\nThe player hits card : {player_table_cards[-1]}")

                    print("\nPlayer's hand :")
                    # using list comprehension to print cards on table
                    [print(i) for i in player_table_cards]
                    print()

                    player_cards_val = 0

                    for i in player_table_cards:
                        player_cards_val += i.value

                    if player_cards_val == 21:
                        print("You got a blackjack!")
                        break

                    elif player_cards_val < 21:  # looping the options again
                        continue

                    else:
                        print("YOU BUSTED!")
                        break

                elif hit_or_stand == "stand":
                    player_cards_val = 0
                    for i in player_table_cards:
                        player_cards_val += i.value

                    print("\nPlayer has decided to stand.")

                    print("\nPlayer's hand:")
                    [print(i) for i in player_table_cards]
                    print()

                    if player_cards_val == 21:
                        print("Player got a blackjack!")
                        break

                    break

                else:
                    print("Enter a valid option. \n")
                    continue

            # variable that stores how many times dealer hits before its cards value is more than equal to 17
            no_of_hits = 0

            while True:
                dealer_cards_val = 0

                for i in dealer_table_cards:  # updating value of dealer's cards
                    dealer_cards_val += i.value

                if dealer_cards_val < 17:
                    no_of_hits += 1
                    dealer_table_cards.append(new_deck.deal_one())
                    continue

                elif 17 <= dealer_cards_val < 21:
                    print(f"The Dealer has hit {no_of_hits} times.")
                    print("\nDealer's hand :")
                    [print(i) for i in dealer_table_cards]
                    break

                elif dealer_cards_val == 21:
                    print(f"The Dealer has hit {no_of_hits} times.")
                    print("The Dealer got a blackjack!")
                    print("\nDealer's hand :")
                    [print(i) for i in dealer_table_cards]
                    break

                elif dealer_cards_val > 21:
                    # checking if player has also busted or not. If player busts , dealer's bust doesn't count.
                    if not (player_cards_val > 21):
                        print(f"The Dealer has hit {no_of_hits} times.")
                        print("The Dealer busted!")
                        print("\nDealer's hand :")
                        [print(i) for i in dealer_table_cards]
                        break

                    else:
                        break

            # checking for busts first
            if player_cards_val > 21:
                print(
                    "\nSince the Player busted , the round is lost.\nPlayer lost the bet money"
                )

            elif dealer_cards_val > 21:
                print(
                    "\n Since the Dealer busted , Player won the round! \nPlayer got twice the money bet."
                )
                chips += bet * 2

            # checking for player's blackjack then
            elif player_cards_val == 21:
                print("\nPlayer won with a blackjack! \nPlayer got thrice the money bet.")
                chips += bet * 3

            # checking whose value is closer to 21
            elif 21 - player_cards_val > 21 - dealer_cards_val:
                print("\nDealer won the round. \nPlayer lost the bet money")

            elif 21 - dealer_cards_val > 21 - player_cards_val:
                print("\nPlayer won the round. \nPlayer got twice the money bet.")
                chips += bet * 2

            # last situation can only be a tie
            else:
                print("\nIt's a tie. \nBet money was returned.")
                chips += bet

            if chips == 0:
                print("\nYou are out of chips , Game over.")
                break

            else:
                cont = input("Do you want to continue? (y/n) :")
                check = cont.upper()  ###So a capital or lowercase value can be entered

                if check == "Y":
                    print("\n" * 100)

                    print(BLACKJACK_STR)

                    continue

                else:
                    print(f"\nTotal amount of chips left with the player = {chips}")
                    print(input("Press Enter to exit the terminal..."))
                    break

        except Exception as error:
            print(f"Following error occurred : {error} \nPlease try again.")
            game_num -= 1  # round with error won't be counted
            continue

if __name__ == "__main__":
    main()